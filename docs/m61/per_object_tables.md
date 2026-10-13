# M6.1 객체별·영상별 표 (J&F, P5 = B1 대비 Δ)

- **포함 기준:** 작은 객체(< 1%), TPLR 적용 객체, 또는 A2x Δ가 0.005를 넘는 객체만 객체 표에 실었다.
- **빈 칸:** 해당 arm을 그 세트에서 실행하지 않았다는 뜻이다.
  - F2와 head는 DAVIS val + PUMaVOS B만 실행했다.
  - A3s와 T1은 crop 적용 21개 객체만 실행했다.
- **crop 미적용 객체:** A2x, A2c, A3는 규칙상 B1과 같아 Δ = 0이다.
- **원자료:** `outputs/milestone61/comparison_per_unit.csv`, `trackA/*.csv`

## 객체별

| 세트 | 객체 | 면적% | B1 J&F | ΔB0 | ΔA2x | ΔA2c | ΔA3 | ΔA3s | ΔT1 | ΔF2 | Δhead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DAVIS_val | davis/lab-coat/obj2 | 0.003 | 0.707 | -0.707 | +0.022 | +0.013 | +0.043 | +0.062 | +0.007 | -0.011 | -0.017 |
| DAVIS_val | davis/lab-coat/obj1 | 0.004 | 0.750 | -0.017 | +0.060 | +0.021 | +0.064 | -0.531 | -0.007 | +0.008 | -0.020 |
| DAVIS_val | davis/kite-surf/obj2 | 0.083 | 0.451 | -0.002 | +0.024 | +0.024 | +0.024 | +0.024 | -0.000 | +0.003 | +0.006 |
| DAVIS_val | davis/soapbox/obj2 | 0.311 | 0.925 | -0.001 | +0.029 | +0.012 | +0.029 | +0.029 | -0.004 | -0.002 | -0.009 |
| DAVIS_val | davis/soapbox/obj3 | 0.311 | 0.930 | +0.000 | +0.019 | +0.015 | +0.020 | +0.020 | +0.007 | +0.001 | -0.004 |
| DAVIS_val | davis/scooter-black/obj1 | 0.326 | 0.924 | -0.001 | +0.010 | +0.011 | +0.010 | +0.010 | -0.008 | -0.001 | -0.006 |
| DAVIS_val | davis/drift-chicane/obj1 | 0.578 | 0.948 | -0.001 | +0.012 | +0.011 | +0.013 | +0.013 | +0.002 | +0.003 | -0.000 |
| DAVIS_val | davis/mbike-trick/obj1 | 0.598 | 0.900 | -0.002 | +0.026 | +0.012 | +0.026 | +0.026 | +0.002 | -0.002 | -0.009 |
| DAVIS_val | davis/mbike-trick/obj2 | 0.651 | 0.870 | -0.019 | +0.026 | +0.005 | +0.032 | +0.032 | +0.007 | -0.001 | -0.004 |
| DAVIS_val | davis/kite-surf/obj1 | 0.817 | 0.657 | -0.214 | +0.000 | +0.000 | +0.000 |  |  | +0.003 | +0.016 |
| DAVIS_val | davis/dogs-jump/obj1 | 0.866 | 0.952 | -0.000 | +0.006 | +0.004 | +0.006 | +0.006 | -0.005 | -0.003 | +0.000 |
| DAVIS_val | davis/paragliding-launch/obj3 | 0.927 | 0.425 | -0.073 | +0.000 | +0.000 | +0.000 |  |  | -0.005 | -0.004 |
| DAVIS_val | davis/soapbox/obj1 | 0.943 | 0.932 | +0.000 | +0.011 | +0.006 | +0.011 | +0.011 | -0.055 | +0.000 | -0.006 |
| DAVIS_val | davis/drift-straight/obj1 | 1.156 | 0.973 | -0.000 | +0.002 | +0.002 | +0.002 | +0.004 | -0.070 | +0.001 | -0.004 |
| DAVIS_val | davis/motocross-jump/obj2 | 1.309 | 0.872 | -0.001 | +0.002 | -0.001 | +0.002 | +0.006 | -0.159 | +0.001 | -0.011 |
| DAVIS_val | davis/scooter-black/obj2 | 1.489 | 0.918 | -0.001 | +0.010 | +0.003 | +0.010 | +0.010 | -0.017 | +0.000 | -0.005 |
| PUMaVOS_A | pumavos/pimples/obj1 | 0.055 | 0.705 | -0.001 | +0.000 | +0.000 | +0.000 |  |  |  |  |
| PUMaVOS_A | pumavos/smile/obj1 | 0.342 | 0.921 | +0.002 | +0.008 | +0.010 | +0.010 | +0.010 | -0.010 |  |  |
| PUMaVOS_A | pumavos/eye_area/obj1 | 0.670 | 0.864 | -0.001 | +0.006 | +0.006 | +0.006 | +0.006 | -0.299 |  |  |
| PUMaVOS_B | pumavos/ice_cream/obj1 | 0.485 | 0.947 | -0.001 | +0.006 | +0.006 | +0.006 | +0.006 | +0.005 | +0.002 | -0.008 |
| PUMaVOS_B | pumavos/queen_car/obj1 | 0.801 | 0.940 | -0.001 | +0.012 | +0.008 | +0.013 | +0.012 | -0.009 | +0.004 | -0.003 |
| SegTrack | segtrack/birdfall/obj1 | 0.397 | 0.884 | +0.000 | -0.023 | -0.023 | -0.023 | -0.023 | -0.034 |  |  |
| SegTrack | segtrack/cheetah/obj1 | 1.240 | 0.900 | +0.000 | +0.013 | +0.011 | +0.013 | +0.013 | -0.021 |  |  |
| SegTrack | segtrack/parachute/obj1 | 2.278 | 0.965 | -0.002 | +0.005 | +0.005 | +0.005 | +0.005 | -0.001 |  |  |

## 영상별 (객체 평균)

| 세트 | 영상 | B1 | ΔA2x | ΔA3 | ΔF2 | Δhead |
|---|---|---|---|---|---|---|
| DAVIS_val | davis/bike-packing | 0.884 | +0.000 | +0.000 | -0.008 | -0.008 |
| DAVIS_val | davis/blackswan | 0.978 | +0.000 | +0.000 | +0.001 | +0.001 |
| DAVIS_val | davis/bmx-trees | 0.812 | +0.000 | +0.000 | +0.001 | -0.004 |
| DAVIS_val | davis/breakdance | 0.967 | +0.000 | +0.000 | +0.001 | +0.001 |
| DAVIS_val | davis/camel | 0.986 | +0.000 | +0.000 | -0.000 | -0.002 |
| DAVIS_val | davis/car-roundabout | 0.981 | +0.000 | +0.000 | -0.000 | -0.001 |
| DAVIS_val | davis/car-shadow | 0.987 | +0.000 | +0.000 | +0.000 | -0.001 |
| DAVIS_val | davis/cows | 0.981 | +0.000 | +0.000 | +0.001 | -0.003 |
| DAVIS_val | davis/dance-twirl | 0.949 | +0.000 | +0.000 | -0.001 | +0.001 |
| DAVIS_val | davis/dog | 0.976 | +0.000 | +0.000 | +0.001 | +0.001 |
| DAVIS_val | davis/dogs-jump | 0.963 | +0.002 | +0.002 | -0.000 | -0.002 |
| DAVIS_val | davis/drift-chicane | 0.948 | +0.012 | +0.013 | +0.003 | -0.000 |
| DAVIS_val | davis/drift-straight | 0.973 | +0.002 | +0.002 | +0.001 | -0.004 |
| DAVIS_val | davis/goat | 0.950 | +0.000 | +0.000 | +0.001 | -0.007 |
| DAVIS_val | davis/gold-fish | 0.909 | +0.000 | +0.000 | +0.005 | -0.005 |
| DAVIS_val | davis/horsejump-high | 0.922 | +0.000 | +0.000 | -0.000 | +0.005 |
| DAVIS_val | davis/india | 0.856 | +0.000 | +0.000 | -0.007 | -0.005 |
| DAVIS_val | davis/judo | 0.870 | +0.000 | +0.000 | +0.012 | -0.001 |
| DAVIS_val | davis/kite-surf | 0.633 | +0.008 | +0.008 | +0.003 | +0.007 |
| DAVIS_val | davis/lab-coat | 0.856 | +0.016 | +0.021 | +0.001 | -0.005 |
| DAVIS_val | davis/libby | 0.955 | +0.000 | +0.000 | -0.002 | -0.005 |
| DAVIS_val | davis/loading | 0.961 | +0.000 | +0.000 | +0.000 | -0.002 |
| DAVIS_val | davis/mbike-trick | 0.885 | +0.026 | +0.029 | -0.001 | -0.007 |
| DAVIS_val | davis/motocross-jump | 0.906 | +0.001 | +0.001 | -0.003 | -0.006 |
| DAVIS_val | davis/paragliding-launch | 0.696 | +0.000 | +0.000 | +0.006 | -0.001 |
| DAVIS_val | davis/parkour | 0.972 | +0.000 | +0.000 | +0.000 | +0.003 |
| DAVIS_val | davis/pigs | 0.937 | +0.000 | +0.000 | +0.005 | -0.006 |
| DAVIS_val | davis/scooter-black | 0.921 | +0.010 | +0.010 | -0.000 | -0.005 |
| DAVIS_val | davis/shooting | 0.925 | +0.000 | +0.000 | -0.000 | -0.012 |
| DAVIS_val | davis/soapbox | 0.929 | +0.020 | +0.020 | -0.000 | -0.006 |
| PUMaVOS_A | pumavos/billie_shoes | 0.580 | +0.000 | +0.000 |  | +0.000 |
| PUMaVOS_A | pumavos/chair | 0.960 | +0.000 | +0.000 |  |  |
| PUMaVOS_A | pumavos/couple_1_face | 0.964 | +0.000 | +0.000 |  |  |
| PUMaVOS_A | pumavos/eye_area | 0.864 | +0.006 | +0.006 |  |  |
| PUMaVOS_A | pumavos/half_face_1 | 0.914 | +0.000 | +0.000 |  |  |
| PUMaVOS_A | pumavos/half_face_2 | 0.867 | +0.000 | +0.000 |  |  |
| PUMaVOS_A | pumavos/pants_workout | 0.985 | +0.000 | +0.000 |  |  |
| PUMaVOS_A | pumavos/pimples | 0.705 | +0.000 | +0.000 |  |  |
| PUMaVOS_A | pumavos/smile | 0.921 | +0.008 | +0.010 |  |  |
| PUMaVOS_B | pumavos/full_face_1 | 0.986 | +0.000 | +0.000 | +0.001 | -0.002 |
| PUMaVOS_B | pumavos/full_face_2 | 0.983 | +0.000 | +0.000 | +0.001 | -0.005 |
| PUMaVOS_B | pumavos/guitar | 0.662 | +0.000 | +0.000 | +0.002 | -0.000 |
| PUMaVOS_B | pumavos/ice_cream | 0.947 | +0.006 | +0.006 | +0.002 | -0.008 |
| PUMaVOS_B | pumavos/queen_car | 0.940 | +0.012 | +0.013 | +0.004 | -0.003 |
| PUMaVOS_B | pumavos/skz | 0.955 | +0.000 | +0.000 | -0.000 | -0.020 |
| SegTrack | segtrack/bird_of_paradise | 0.947 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/birdfall | 0.884 | -0.023 | -0.023 |  |  |
| SegTrack | segtrack/bmx | 0.860 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/cheetah | 0.891 | +0.006 | +0.006 |  |  |
| SegTrack | segtrack/drift | 0.950 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/frog | 0.951 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/girl | 0.963 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/hummingbird | 0.911 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/monkey | 0.940 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/monkeydog | 0.923 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/parachute | 0.965 | +0.005 | +0.005 |  |  |
| SegTrack | segtrack/penguin | 0.980 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/soldier | 0.950 | +0.000 | +0.000 |  |  |
| SegTrack | segtrack/worm | 0.765 | +0.000 | +0.000 |  |  |
