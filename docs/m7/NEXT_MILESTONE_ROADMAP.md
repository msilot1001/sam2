# 다음 milestone 로드맵 (M7.1 / M8)

**출발점:**
- `exp/m7-algorithm-research`, FINAL_DECISION: KEEP_SGGL
- 코드: `src/spatial/sggl.py`, `policies.py`, `scripts/m7.py`
- GPU: Vast RTX 5090. Git 원본과 영구 보관은 Ubuntu.

**M7이 남긴 가장 큰 미해결 문제와 근거:**
1. **두 view의 동시 부재:** 감사상 SGGL 손실 1위(156.6 frame-JF, 영상 14개).
2. **데이터 부족:** 극소 대상 7객체, 거짓 재획득 1 영상 등 표본이 작다.
3. **M3-vel의 검정력 부족:** 이득 영상 2개.

## 작업 목록 (각 작업은 사전 등록 → 실행 → 같은 기계 기준선 비교)

| ID | 목표 | 파일 | 수용 기준 |
|---|---|---|---|
| **M71-01** | 극소 대상 데이터 확보 **(사용자 승인 필요)**. 후보: LVOS v2 val, SA-V val/test, VOST. 라이선스, 크기, SAM 2 학습 겹침을 확인하고, GT만으로 prompt 면적 < 0.1% 객체를 센다(SAM 2 실행 금지) | `scripts/m71_data.py`, `docs/m71/DATA.md` | 극소 객체 ≥ 20개, 영상 ≥ 12개, 거짓 재획득 후보 사건 ≥ 5 영상. 영상 단위 분할을 추론 전에 고정 |
| M71-02 | VOTS2025 참가 추적기 설명 읽기 (crop + 전체 frame + 검증 선례) | `docs/m71/PRIOR_ART_VOTS.md` | 직접 선례 유무를 기록. 있으면 주장 범위 축소 |
| M71-03 | 새 데이터 개발 절반에서 SGGL, SGGL-V, M3-vel, P5, B0 실행 (Vast, 4 worker × 3 thread) | `scripts/m7.py run` | 기준: SGGL vs P5 영상 CI 하한 > 0. M3-vel은 사전 pilot 규칙 ADVANCE일 때만 채택 |
| M71-04 | '두 view 동시 부재' 전용 후보 생성: 부재 연속 k frame 뒤 L2(2× crop) 또는 전역 재탐지 후보를 만들고, 신원 + G 일치로 확인. M7 재생에서 L2 fallback은 +3.4 frame-JF였고, miami-surf에서는 −14 | `src/spatial/policies.py` | 거짓 재획득 사건에서 손실 증가 없이 부재 frame 회수율 향상. 사전 pilot 규칙 |
| M71-05 | (조건부) T head를 대규모 데이터(SA-V train 일부)로 학습. 단 M71-01 승인 시만. M7은 18 영상에서 일반화하지 않았다 | `src/spatial/feature_fusion.py` | 학습에 쓰지 않은 영상에서 J > 입력 J, 그리고 DEV 영상 평가 > SGGL |
| M71-06 | 확인 1회: 코드를 고정하고 커밋한 뒤, 새 데이터 확인 절반 + PUMaVOS C (C는 Ubuntu에서만 존재. 실행 시 그 영상만 Vast로 전송하거나 Ubuntu 2080에서 실행) | `scripts/m7.py confirm` (신설) | 결과를 모두 보고하고 이후 튜닝하지 않는다 |
| M71-07 | 원고: 상한 분해, 실패 분류, 해상도 probe 발견, 음성 결과, VK 재현 | `docs/m71/` | 초안 |

## 계산 예산 (RTX 5090, 이번 M7에서 측정한 처리량 기준)

**처리량:**
- 4 worker × 3 thread 기준 약 8.8 frame/s(집계).
- RTX 2080은 약 2.4 frame/s(GPU 제한). 독립 작업이나 확인 실행에 쓴다.

**새 데이터 1만 frame 기준 추정:**

| 작업 | 소요 |
|---|---|
| arm 5개 | 약 1.6시간 |
| 재생 / 분석 | CPU |
| T 학습 (GPU 상주 데이터) | head당 약 1분 |

**운영 규칙:**
- Vast 기본값: `OMP_NUM_THREADS=3–4`.
- 학습 캐시는 GPU에 상주시킨다(`M7_TDATA_GPU=1`, 기본값).
- 병렬 실행은 `M7_SHARD=i/n`으로 한다.
