# M8 병렬 실행 계획

## 1. 작업 의존 그래프

```
prereg ──► diag (Stage 1, 3 shard) ──► diag1b (shard-0 units only, amendment 1)
                │                              │
refs (frame-0 tokens) ─────────────────────────┤
                                               ▼
                                 m8_analysis stage1 + rankers (CPU)
                                               │
                         arms.json freeze (Stage 2 arms, REP-fitted thresholds)
                                               │
              ┌────────────────┬───────────────┼───────────────┬──────────────┐
           arm 1            arm 2           arm 3          M3-vel ctrl    (each: 3 shards)
              └────────────────┴───────────────┼───────────────┴──────────────┘
                                               ▼
                         m8_analysis causal (CPU) ──► Stage 3 combos ──► VK finalist (Vast) ──► docs, review
```

## 2. 작업 종류별 배치 원칙

| 작업 | 성격 | 배치 |
|---|---|---|
| diag / 인과 arm (SAM 2 rollout) | GPU-bound, 영상 단위로 독립 | 단위(shard) 단위로 기계에 고정. **같은 단위의 모든 arm과 기준선은 같은 기계**에서 돈다 (machine-matched 비교) |
| frame-0 token, ranker 학습 | 작음 | 2080 또는 CPU |
| 통계, bootstrap, 보고서, 그림 | CPU | Ubuntu (Vast CPU quota를 쓰지 않음) |
| 선행연구 감사 | 네트워크 | 백그라운드 agent |

## 3. 기계별 계획과 실제

- **RTX 2080 (Ubuntu):** M7 측정으로 GPU-bound(≈2.4 frame/s, worker 수와 무관). 1 worker. diag shard 0→1→2를 연쇄 실행(`outputs/milestone8/chain_diag.sh`). Vast가 오면 남은 shard를 넘긴다(`STOP_UBUNTU_DIAG_<i>` 파일로 건너뜀).
- **RTX 5090 (Vast):** M7 측정: 4 worker × 3 thread가 가장 좋았다(8.8 frame/s). M8 작업은 trigger frame에서 probe가 25–31개라 작업 형태가 다르므로, 도착하면 1/2/4 worker를 다시 측정한다.
- **실제 경과:**
  - Vast(89.221.67.161:16328)는 M8 시작부터 연결이 거부되었다.
  - 17:00 KST부터 사용자가 RunPod RTX 4090 워커를 제공했다. 실제 CPU quota는 13.6 vCPU이고 RAM은 251 GB다.
  - 운영 원칙은 [`../GPU_WORKER_OPERATIONS.md`](../GPU_WORKER_OPERATIONS.md)와 `AGENTS.md`를 따른다. Ubuntu가 유일한 원본이다.

### 3.1 RunPod 이전 계획 (17:00 KST 결정)

| 작업 | 기계 | 이유 |
|---|---|---|
| diag shard 1, 2 (진행 중), diag1b (shard 0) | **Ubuntu 2080 그대로** | 이미 실행 중이며 중단하지 않는다. 단위별 기준선이 이 기계에 있다 |
| Stage 1 분석, ranker 학습(R2/R3), 보고서, 그림 | Ubuntu CPU | 완료된 CPU 결과를 그대로 유지한다 |
| 인과 arm: `SGGL`(같은 기계 기준선), `M3vel`, `FLOWVEL`, Stage 2/3 rescue arm | **RunPod 4090** | 모든 arm과 그 기준선을 같은 기계에서 돌린다(machine-matched). 출력은 `outputs/milestone8/runpod/runs/<arm>/` |
| VK 결선 | RunPod | 결선 후보가 정해진 뒤 |
| 결과 회수 | Ubuntu tmux `m8:sync` → `gpu_sync.py loop` | 검증된 pull만 원본으로 인정한다 |

**재현성 확인:** RunPod `SGGL` arm을 Ubuntu diag의 SGGL과 M7 Vast SGGL에 단위별로 비교한다. 차이가 있으면 기록하고, 판정은 RunPod 기준선 안에서만 한다.

## 4. 재개 방법

- 모든 명령은 단위 CSV가 이미 있으면 건너뛴다(원자적 쓰기). 중단해도 같은 명령으로 재개된다.
- `M7_SHARD=i/n`은 단위 순서가 고정된 결정적 분할이다. 기계 이력은 각 run 디렉터리의 `PROVENANCE.json`에 남긴다.
