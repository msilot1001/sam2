# 그림 6 예시 단위 선택 근거

`scripts/paper_fig6.py select`가 계산했다(손으로 고르지 않음). 원자료: `outputs/milestone10/runpod/runs/{B0,OR0}` (RunPod 4090 dev 행).

## 규칙

```
- MOSEv2: tiny unit (area0 < 0.001) with the largest unit-mean SGGL - B0 J&F (paired frames, 4090 rows).
- VOST: unit with the largest (SGGL - B0) per-frame loss contribution of wrong_object frames
  (sum of 1 - J&F over wrong_object frames / frames, as amend9 Table 3).
- Frames: in the WORSE arm of that unit (MOSEv2: B0, VOST: SGGL), loss episodes = maximal runs of GT-present frames with
  J < 0.5 (m10_analysis.episodes); take the episode with the largest summed J&F gap (better - worse arm). Show
  the last GT-present frame with J >= 0.5 before the onset (the onset itself if none: the arm fails from frame 1),
  the episode's middle frame, and the first GT-present frame with J >= 0.5 after it (recovery) or the unit's last frame.
  (Revised once after the first draft picked a GT-absent frame as 'before'; disclosed in fig6_selection.md.)
```

## 선택

| 데이터셋 | 단위 | area0 | B0 J&F | SGGL J&F | 에피소드를 잡은 arm | 에피소드 (시작, 길이, 회복) | 그린 frame (video_idx) |
|---|---|---|---|---|---|---|---|
| mosev2 | `mosev2__mosev2/2c713560__obj1` | 0.00001 | 0.000 | 0.876 | B0 | 1, 62 frame, 끝까지 | 1, 31, 62 |
| vost | `vost__vost/1186_cut_chilli__obj2` | 0.00067 | 0.320 | 0.294 | SGGL | 55, 33 frame, 구간 끊김 | 50, 71, 157 |

## 후보 순위 (상위 5)

MOSEv2 극소 단위, SGGL − B0 (단위 평균 J&F):

| 단위 | SGGL − B0 | area0 |
|---|---|---|
| `mosev2__mosev2/2c713560__obj1` | +0.876 | 0.00001 |
| `mosev2__mosev2/82ucmhmx__obj2` | +0.774 | 0.00037 |
| `mosev2__mosev2/82ucmhmx__obj1` | +0.576 | 0.00039 |
| `mosev2__mosev2/5ten05az__obj1` | +0.441 | 0.00068 |
| `mosev2__mosev2/fa3e0b85__obj7` | +0.362 | 0.00042 |

VOST, wrong_object frame 손실 기여 (SGGL − B0):

| 단위 | 차이 |
|---|---|
| `vost__vost/1186_cut_chilli__obj2` | +0.391 |
| `vost__vost/9673_cut_tomato__obj1` | +0.232 |
| `vost__vost/1201_cut_chili__obj1` | +0.159 |
| `vost__vost/3559_break_egg__obj2` | +0.059 |
| `vost__vost/1206_cut_mango__obj1` | +0.040 |

## 마스크 재실행 (Ubuntu RTX 2080)

그림의 마스크는 2080에서 다시 돌린 것이다. 표의 수치는 4090 행이다. 두 기계의 단위 평균 J&F:

| 단위 | arm | 4090 | 2080 | 차이 |
|---|---|---|---|---|
| `mosev2__mosev2/2c713560__obj1` | B0 | 0.0000 | 0.0000 | +0.0000 |
| `mosev2__mosev2/2c713560__obj1` | SGGL | 0.8762 | 0.8762 | +0.0000 |
| `vost__vost/1186_cut_chilli__obj2` | B0 | 0.3201 | 0.3201 | +0.0000 |
| `vost__vost/1186_cut_chilli__obj2` | SGGL | 0.2938 | 0.2938 | +0.0000 |

## 공개

- frame 규칙은 한 번 고쳤다. 첫 초안이 '직전' frame으로 GT 없는 frame(VOST t=54)을 골라, 'GT가 있고 J ≥ 0.5인 가장 가까운 frame'으로 바꿨다. 단위 선택 규칙은 바꾸지 않았다.
- MOSEv2는 B0가 frame 1부터 끝까지 놓친다(에피소드 1개, 회복 없음). 그래서 '직전' 자리에 시작 frame을, '회복' 자리에 마지막 frame을 그렸다.
- VOST의 손실 구간은 GT 없는 frame에서 끊기지만(표의 '구간 끊김'), 그 뒤 SGGL이 GT 있고 J ≥ 0.5로 돌아온 frame이 없어 세 번째 자리에 마지막 frame을 그렸다. 즉 회복하지 못했다.
- MOSEv2 영상 `2c713560`의 frame은 Ubuntu에 없어, 공개 HF `FudanCVL/MOSEv2` train 분할 3개를 한 번 스트리밍해 이 영상만 추출했다. 분할 sha256은 M9 기록(`outputs/milestone9/manifests/mosev2_download_sha256.txt`)과 일치한다(`outputs/milestone10/fig6/dl/`).
- 패널 위 J&F는 2080 재실행 값이다. 극소 객체가 보이도록 행마다 확대했다(마스크·GT·crop 사각형을 모두 담는 영역).
