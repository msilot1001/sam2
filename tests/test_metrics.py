import numpy as np
import pytest

from baseline.metrics import dilate, disk_offsets, f_measure, j_measure, seg2bmap


def box(shape=(60, 80), y=(10, 30), x=(20, 50)):
    m = np.zeros(shape, bool)
    m[y[0]:y[1], x[0]:x[1]] = True
    return m


def test_j_values():
    a, b = box(), box(x=(35, 65))
    inter, union = 20 * 15, 20 * 45
    assert j_measure(a, b) == pytest.approx(inter / union)
    assert j_measure(a, a) == 1.0
    assert j_measure(a, ~a) == 0.0


def test_empty_mask_conventions():
    z = np.zeros((60, 80), bool)
    assert j_measure(z, z) == 1.0 and f_measure(z, z) == 1.0          # both empty: correct "absent"
    assert j_measure(z, box()) == 0.0 and f_measure(z, box()) == 0.0  # missed object
    assert j_measure(box(), z) == 0.0 and f_measure(box(), z) == 0.0  # false positive on empty GT


def test_seg2bmap_square():
    m = box((10, 10), (2, 6), (3, 8))  # rows 2..5, cols 3..7
    b = seg2bmap(m)
    # A pixel is boundary if it differs from its right, lower or lower-right neighbour: the OUTER
    # ring on the top/left (row 1 cols 2..7, col 2 rows 2..5) and the INNER ring on the
    # bottom/right (col 7 rows 2..5, row 5 cols 3..6) -> 6 + 4 + 4 + 4 = 18 pixels.
    assert b.sum() == 18
    assert b[1, 2] and b[1, 7] and b[2, 2] and b[2, 7] and b[5, 3] and b[5, 7]
    assert not b[2, 3] and not b[4, 6] and not b[1, 8] and not b[6, 7]
    assert seg2bmap(np.zeros((5, 5), bool)).sum() == 0


def test_dilate_matches_bruteforce():
    rng = np.random.default_rng(0)
    m = rng.random((30, 40)) > 0.97
    r = 3
    brute = np.zeros_like(m)
    ys, xs = np.nonzero(m)
    for y in range(30):
        for x in range(40):
            brute[y, x] = np.any((ys - y) ** 2 + (xs - x) ** 2 <= r * r)
    np.testing.assert_array_equal(dilate(m, r), brute)
    assert len(disk_offsets(1)) == 5


def test_f_tolerance():
    a = box((300, 400), (100, 200), (100, 250))
    tol = int(np.ceil(0.008 * np.hypot(300, 400)))  # 4 px
    assert f_measure(a, a) == 1.0
    near = np.roll(a, tol - 1, axis=1)
    far = np.roll(a, 3 * tol, axis=1)
    assert f_measure(near, a) == 1.0               # within tolerance -> every boundary pixel matches
    assert 0 < f_measure(far, a) < 0.8              # vertical edges miss, horizontal ones still match
    assert f_measure(a, a, bound_th=2) == 1.0       # absolute pixel tolerance form


def test_shape_mismatch():
    with pytest.raises(ValueError):
        j_measure(np.zeros((2, 2), bool), np.zeros((3, 2), bool))
