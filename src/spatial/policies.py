"""M7 candidate policies over one frame's global/local observations (replay or live; causal; no GT).

An observation `o` (dict) for frame t holds frame-res logits lG (H,W), lL / lL2 / lLp / lLt (NaN outside their
crop), masks mG, mL (pasted), mL_view (crop res), scores sG, sL, sL2, sLp, sLt, touchL, box, simL (64x64 cosine map
in the crop view), hw. A policy returns (output mask, accepted_L) and owns a `Hist`.

Families: S (selection evidence), F (mask fusion on top of SGGL's selection), M (memory-conditioned checks),
R (cross-scale consistency). Names match docs/m7/18_CANDIDATE_MATRIX.md.
"""
import numpy as np

from spatial import evidence, mask_fusion
from spatial.evidence import Hist


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def selfok(o):
    return bool(o["sL"] > 0 and o["mL"].any() and not o["touchL"])


def g_support(o, thr=0.1):
    """G sees a target at the candidate's location."""
    return bool(o["sG"] > 0 and _iou(o["mL"], o["mG"]) > thr)


class Policy:
    name = "SGGL"
    needs = ()                     # probes this policy needs live: subset of {"Lp", "Lt", "L2"}

    def __init__(self, gt0):
        b = gt0.nonzero()
        self.h = Hist(max(b[0].max() - b[0].min() + 1, b[1].max() - b[1].min() + 1))
        self.log = {}

    def accept(self, o) -> bool:
        return selfok(o)

    def output(self, o, acc):
        return o["mL"] if acc else o["mG"]

    def __call__(self, o):
        acc = self.accept(o)
        out = self.output(o, acc)
        s_id, _ = evidence.identity(o["simL"], o["mL_view"]) if o["mL_view"].any() else (None, None)
        self.h.update(out, selfok(o), acc, o["sG"], s_id, g_support(o) or self._confirm(o))
        return out, acc

    def _confirm(self, o):
        return False


class SGGLV(Policy):
    name = "SGGL-V"

    def accept(self, o):
        return selfok(o) and o["sG"] > 0


# ----------------------------------------------------------------------------- S family
class S1Persist(Policy):
    """S1 control: a re-acquired candidate (previous output empty) is accepted only after K consecutive valid frames."""
    name, K = "S1-K3", 3

    def accept(self, o):
        if not selfok(o):
            return False
        return not (self.h.lost or self.h.reacq) or self.h.l_run + 1 >= self.K


class S1State(Policy):
    """S1: state-aware veto. Global support is required only while a re-acquisition is unconfirmed (after the output
    was empty); continuous tracking keeps the plain self gate (unlike SGGL-V's always-on veto)."""
    name = "S1-state"

    def accept(self, o):
        if not selfok(o):
            return False
        return g_support(o) if (self.h.lost or self.h.reacq) else True


class S2Observe(Policy):
    """S2: global absence vetoes the local candidate only when G could observe the target (target side >= 24 px in
    G's input and G recently reliable)."""
    name = "S2-observe"

    def accept(self, o):
        if not selfok(o):
            return False
        return o["sG"] > 0 or not self.h.g_observable(o["hw"])


class S3Identity(Policy):
    """S3 absolute: reject a local candidate whose similarity to the frozen frame-0 reference (stride-16 backbone,
    crop view) falls > 3 sd below its own accepted history."""
    name = "S3-abs"

    def accept(self, o):
        if not selfok(o):
            return False
        s, _ = evidence.identity(o["simL"], o["mL_view"])
        return self.h.id_z(s) >= evidence.Z_MIN


class S3Relative(Policy):
    """S3 relative: reject a local candidate when another crop region is MORE similar to the frozen reference than
    the candidate (negative distractor margin) and G does not support it."""
    name = "S3-rel"

    def accept(self, o):
        if not selfok(o):
            return False
        s, margin = evidence.identity(o["simL"], o["mL_view"])
        return margin >= 0 or g_support(o)


class S4Evidence(Policy):
    """S4: G's verdict is classified as positive / reliable-negative / conflicting / insufficient. Only reliable
    negative evidence (G observable and absent) or a conflict (G present elsewhere) without identity support can
    remove the local candidate; insufficient global evidence never erases it."""
    name = "S4-evidence"

    def accept(self, o):
        if not selfok(o):
            return False
        if g_support(o):
            return True
        obs = self.h.g_observable(o["hw"])
        s, margin = evidence.identity(o["simL"], o["mL_view"])
        id_ok = self.h.id_z(s) >= evidence.Z_MIN and margin >= 0
        if o["sG"] <= 0:
            return (not obs) or id_ok
        return id_ok                                   # G present but elsewhere: identity decides


# ----------------------------------------------------------------------------- F family (selection = SGGL)
class F1Blend(Policy):
    name, wL = "F1-blend", 0.5

    def output(self, o, acc):
        return mask_fusion.blend(o["lL"], o["lG"], self.wL) if acc else o["mG"]


class F1BlendL(F1Blend):
    name, wL = "F1-blend.7", 0.7


class F2Boundary(Policy):
    name = "F2-boundary"

    def output(self, o, acc):
        return mask_fusion.boundary_refine(o["mL"], o["lL"], o["lG"], r=2) if acc else o["mG"]


class F3Asym(Policy):
    name, sym = "F3-asym", False

    def output(self, o, acc):
        if not acc:
            return o["mG"]
        m, a, d = mask_fusion.asymmetric_edit(o["mL"], o["lL"], o["lG"], self.h.g_observable(o["hw"]), symmetric=self.sym)
        self.log = {"added": a, "deleted": d}
        return m


class F3Sym(F3Asym):
    name, sym = "F3-sym", True


class F4Instance(Policy):
    name = "F4-instance"

    def output(self, o, acc):
        return mask_fusion.instance_union(o["mL"], o["mG"]) if acc else o["mG"]


class FUnion(Policy):
    name = "F-union"

    def output(self, o, acc):
        return (o["mL"] | o["mG"]) if acc else o["mG"]


class FInter(Policy):
    name = "F-inter"

    def output(self, o, acc):
        return (o["mL"] & o["mG"]) if acc and o["sG"] > 0 else (o["mL"] if acc else o["mG"])


# ----------------------------------------------------------------------------- M family (protected memory probe)
class M1Verify(Policy):
    """M1: an unconfirmed re-acquisition is accepted if G supports it OR the protected-memory (prompt-only) local
    prediction finds the same candidate (IoU > 0.3, score > 0). Memory-only candidates are rejected."""
    name, needs = "M1-verify", ("Lp",)

    def accept(self, o):
        if not selfok(o):
            return False
        if not (self.h.lost or self.h.reacq):
            return True
        return g_support(o) or self._confirm(o)

    def _confirm(self, o):
        return bool(o["sLp"] > 0 and _iou(o["mL"], o["mLp"]) > 0.3)


class M1Always(Policy):
    """M1 always-on: reject the local candidate whenever the prompt-only memory prediction disagrees with it
    (IoU <= 0.1 or absent) and G does not support it."""
    name, needs = "M1-always", ("Lp",)

    def accept(self, o):
        if not selfok(o):
            return False
        return g_support(o) or (o["sLp"] > 0 and _iou(o["mL"], o["mLp"]) > 0.1)


class M1Trusted(Policy):
    """M1 trusted-memory alternative: in an unconfirmed re-acquisition without G support, output the local prediction
    made from trusted memories only (frames where L and G agreed) instead of the normal one."""
    name, needs = "M1-trusted", ("Lt",)

    def output(self, o, acc):
        if acc and (self.h.lost or self.h.reacq) and not g_support(o):
            return o["mLt"] if o["sLt"] > 0 else o["mG"]
        return o["mL"] if acc else o["mG"]


# ----------------------------------------------------------------------------- M3: crop-centre control (causal only)
class M3Velocity(Policy):
    """M3: output selection = SGGL; the NEXT crop centre is the output's box centre plus half the last centre
    displacement (damped constant velocity). Targets crop escape on fast motion. Changes only the crop trajectory, so it
    cannot be replayed; causal rollout only."""
    name, damp = "M3-vel", 0.5

    def __init__(self, gt0):
        super().__init__(gt0)
        self.prev_c = None

    def center(self, out):
        from spatial import vcam
        c = vcam.center_of(out)
        if c is None:
            self.prev_c = None
            return None
        nxt = c if self.prev_c is None else (c[0] + self.damp * (c[0] - self.prev_c[0]), c[1] + self.damp * (c[1] - self.prev_c[1]))
        self.prev_c = c
        return nxt


class M3VelocityR(M3Velocity):
    """M3 revision (stage 2): M3-vel lost on tiny objects whose box centre jitters or jumps when the output source
    switches between G and L. Velocity is used only when the last two outputs both came from the accepted local crop,
    and the predicted step is capped at side/8 (side = crop side); otherwise the centre is the output's box centre."""
    name = "M3-vel-r"

    def __init__(self, gt0):
        super().__init__(gt0)
        self.prev_acc = False
        self.side = None

    def __call__(self, o):
        out, acc = super().__call__(o)
        self.side = o["box"][1] - o["box"][0]
        self._acc_now, self._acc_prev = acc, self.prev_acc
        self.prev_acc = acc
        return out, acc

    def center(self, out):
        from spatial import vcam
        c = vcam.center_of(out)
        prev, self.prev_c = self.prev_c, c
        if c is None or prev is None or not (self._acc_now and self._acc_prev):
            return c
        cap = self.side / 8.0
        dy = max(-cap, min(cap, self.damp * (c[0] - prev[0])))
        dx = max(-cap, min(cap, self.damp * (c[1] - prev[1])))
        return (c[0] + dy, c[1] + dx)


# ----------------------------------------------------------------------------- K: targeted combinations (stage 3)
def _id_ok(pol, o):
    s, margin = evidence.identity(o["simL"], o["mL_view"])
    return s is not None and pol.h.id_z(s) >= evidence.Z_MIN and margin >= 0


class K1ObsId(Policy):
    """K1 = S2 + S3: global absence vetoes the local candidate only if G could observe the target AND the candidate's
    own identity evidence is weak."""
    name = "K1-obs+id"

    def accept(self, o):
        if not selfok(o):
            return False
        return o["sG"] > 0 or not self.h.g_observable(o["hw"]) or _id_ok(self, o)


class K2StateId(Policy):
    """K2 = S1 + S3: an unconfirmed re-acquisition is accepted with G support OR identity support."""
    name = "K2-state+id"

    def accept(self, o):
        if not selfok(o):
            return False
        return True if not (self.h.lost or self.h.reacq) else (g_support(o) or _id_ok(self, o))

    def _confirm(self, o):
        return _id_ok(self, o)


class K3AsymId(F3Asym):
    """K3 = F3 + S3: asymmetric edits only when the selected local candidate passes the identity check."""
    name = "K3-asym+id"

    def output(self, o, acc):
        return super().output(o, acc) if acc and _id_ok(self, o) else (o["mL"] if acc else o["mG"])


class K5MemObs(Policy):
    """K5 = M1 + S2: the protected-memory disagreement can reject the local candidate only when G is observable and
    absent (memory dependence used only where global negative evidence is reliable)."""
    name, needs = "K5-mem+obs", ("Lp",)

    def accept(self, o):
        if not selfok(o):
            return False
        if o["sG"] > 0 or not self.h.g_observable(o["hw"]):
            return True
        return bool(o["sLp"] > 0 and _iou(o["mL"], o["mLp"]) > 0.1)


COMBOS = [K1ObsId, K2StateId, K3AsymId, K5MemObs]


# ----------------------------------------------------------------------------- R family
class R2Scale(Policy):
    """R2: an unconfirmed re-acquisition needs G support OR agreement of the 2x-wider crop state (cross-scale
    consistency, independent memory)."""
    name, needs = "R2-scale", ("L2",)

    def accept(self, o):
        if not selfok(o):
            return False
        if not (self.h.lost or self.h.reacq):
            return True
        return g_support(o) or self._confirm(o)

    def _confirm(self, o):
        return bool(o["sL2"] > 0 and _iou(o["mL"], o["mL2"]) > 0.3)


ALL = [Policy, SGGLV, S1Persist, S1State, S2Observe, S3Identity, S3Relative, S4Evidence,
       F1Blend, F1BlendL, F2Boundary, F3Asym, F3Sym, F4Instance, FUnion, FInter, M1Verify, M1Always, M1Trusted, R2Scale]
CAUSAL_ONLY = [M3Velocity, M3VelocityR]
