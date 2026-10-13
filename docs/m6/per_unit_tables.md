# M6 객체별 표 (J&F, B1 대비 Δ)

- 빈 칸은 그 arm이 해당 객체에 crop을 적용하지 않았다는 뜻이다(규칙상 B1과 같아 Δ = 0).
- crop이 한 arm에라도 적용된 객체만 실었다. 나머지 객체는 모든 arm이 B1과 같다.
- 원자료: `outputs/milestone6/reports/trackA_*_per_unit.csv`, `S1_per_unit.csv`

| set | object | small | B1 J&F | ΔA1 | ΔA1z | ΔA1v | ΔA2x | ΔS1 | S1 pick |
|---|---|---|---|---|---|---|---|---|---|
| DAVIS_val | davis/bmx-trees/obj1 |  | 0.755 |  | +0.025 |  |  | +0.023 | A1z |
| DAVIS_val | davis/bmx-trees/obj2 |  | 0.869 |  | +0.018 |  |  | +0.017 | A1z |
| DAVIS_val | davis/dogs-jump/obj1 | ✓ | 0.952 | +0.006 | -0.055 | +0.006 | +0.006 | -0.056 | A1z |
| DAVIS_val | davis/dogs-jump/obj2 |  | 0.968 |  | -0.019 |  |  | -0.019 | A1z |
| DAVIS_val | davis/dogs-jump/obj3 |  | 0.970 |  | +0.004 |  |  | +0.004 | A1z |
| DAVIS_val | davis/drift-chicane/obj1 | ✓ | 0.948 | +0.010 | -0.055 | +0.001 | +0.012 | +0.009 | A1 |
| DAVIS_val | davis/drift-straight/obj1 |  | 0.973 | -0.089 | -0.293 | -0.050 | +0.002 | -0.294 | A1z |
| DAVIS_val | davis/gold-fish/obj2 |  | 0.821 |  | +0.099 |  |  | +0.098 | A1z |
| DAVIS_val | davis/gold-fish/obj3 |  | 0.904 |  | +0.012 |  |  | +0.012 | A1z |
| DAVIS_val | davis/horsejump-high/obj2 |  | 0.910 |  | +0.032 |  |  | +0.030 | A1z |
| DAVIS_val | davis/kite-surf/obj2 | ✓ | 0.451 | +0.024 | -0.073 | +0.179 | +0.024 | -0.078 | A1z |
| DAVIS_val | davis/kite-surf/obj3 |  | 0.790 |  | +0.118 |  |  | +0.112 | A1z |
| DAVIS_val | davis/lab-coat/obj1 | ✓ | 0.750 | +0.051 | +0.051 | +0.048 | +0.060 | +0.040 | A1 |
| DAVIS_val | davis/lab-coat/obj2 | ✓ | 0.707 | +0.041 | +0.041 | +0.027 | +0.022 | +0.025 | A1 |
| DAVIS_val | davis/lab-coat/obj3 |  | 0.953 |  | -0.009 |  |  | -0.010 | A1z |
| DAVIS_val | davis/lab-coat/obj4 |  | 0.945 |  | +0.005 |  |  | +0.002 | A1z |
| DAVIS_val | davis/lab-coat/obj5 |  | 0.922 |  | +0.034 |  |  | +0.031 | A1z |
| DAVIS_val | davis/loading/obj2 |  | 0.921 |  | -0.083 |  |  | -0.084 | A1z |
| DAVIS_val | davis/mbike-trick/obj1 | ✓ | 0.900 | +0.026 | -0.017 | +0.024 | +0.026 | -0.019 | A1z |
| DAVIS_val | davis/mbike-trick/obj2 | ✓ | 0.870 | +0.029 | -0.094 | +0.020 | +0.026 | -0.095 | A1z |
| DAVIS_val | davis/motocross-jump/obj1 |  | 0.940 |  | -0.010 |  |  | -0.013 | A1z |
| DAVIS_val | davis/motocross-jump/obj2 |  | 0.872 | -0.114 | -0.563 | -0.082 | +0.002 | -0.557 | A1z |
| DAVIS_val | davis/paragliding-launch/obj1 |  | 0.881 |  | +0.020 |  |  | +0.019 | A1z |
| DAVIS_val | davis/parkour/obj1 |  | 0.972 |  | +0.007 |  |  | +0.007 | A1z |
| DAVIS_val | davis/pigs/obj1 |  | 0.955 |  | +0.021 |  |  | +0.021 | A1z |
| DAVIS_val | davis/scooter-black/obj1 | ✓ | 0.924 | +0.010 | -0.015 | +0.011 | +0.010 | -0.021 | A1z |
| DAVIS_val | davis/scooter-black/obj2 |  | 0.918 | +0.010 | -0.127 | +0.011 | +0.010 | -0.130 | A1z |
| DAVIS_val | davis/soapbox/obj1 | ✓ | 0.932 | -0.015 | -0.225 | -0.004 | +0.011 | -0.227 | A1z |
| DAVIS_val | davis/soapbox/obj2 | ✓ | 0.925 | +0.029 | -0.014 | +0.019 | +0.029 | -0.017 | A1z |
| DAVIS_val | davis/soapbox/obj3 | ✓ | 0.930 | +0.017 | -0.100 | +0.015 | +0.019 | -0.103 | A1z |
| PUMaVOS_A | pumavos/chair/obj1 |  | 0.960 |  | +0.008 |  |  | +0.008 | A1z |
| PUMaVOS_A | pumavos/couple_1_face/obj1 |  | 0.964 |  | -0.006 |  |  | -0.006 | A1z |
| PUMaVOS_A | pumavos/eye_area/obj1 | ✓ | 0.864 | +0.006 | -0.008 | +0.004 | +0.006 | -0.008 | A1z |
| PUMaVOS_A | pumavos/half_face_1/obj1 |  | 0.914 |  | -0.011 |  |  | -0.011 | A1z |
| PUMaVOS_A | pumavos/half_face_2/obj1 |  | 0.867 |  | -0.009 |  |  | -0.009 | A1z |
| PUMaVOS_A | pumavos/pants_workout/obj1 |  | 0.985 |  | +0.004 |  |  | +0.004 | A1z |
| PUMaVOS_A | pumavos/pimples/obj1 | ✓ | 0.705 |  | +0.017 |  |  | +0.017 | A1z |
| PUMaVOS_A | pumavos/smile/obj1 | ✓ | 0.921 | +0.012 | +0.018 | -0.496 | +0.008 | +0.011 | A1 |
| PUMaVOS_B | pumavos/full_face_1/obj1 |  | 0.986 |  | -0.003 |  |  | -0.003 | A1z |
| PUMaVOS_B | pumavos/full_face_2/obj1 |  | 0.983 |  | +0.001 |  |  | +0.001 | A1z |
| PUMaVOS_B | pumavos/ice_cream/obj1 | ✓ | 0.947 | +0.006 | +0.000 | +0.001 | +0.006 | -0.000 | A1z |
| PUMaVOS_B | pumavos/queen_car/obj1 | ✓ | 0.940 | +0.014 | -0.111 | +0.008 | +0.012 | -0.111 | A1z |
| PUMaVOS_B | pumavos/skz/obj1 |  | 0.955 |  | +0.008 |  |  | +0.008 | A1z |
| SegTrack | segtrack/birdfall/obj1 | ✓ | 0.884 | -0.023 | -0.039 | -0.008 | -0.023 | +0.000 | B1 |
| SegTrack | segtrack/bmx/obj2 |  | 0.758 |  | -0.010 |  |  | +0.000 | B1 |
| SegTrack | segtrack/cheetah/obj1 |  | 0.900 | +0.013 | -0.028 | +0.005 | +0.013 | +0.012 | A1 |
| SegTrack | segtrack/cheetah/obj2 |  | 0.883 |  | -0.038 |  |  | +0.000 | B1 |
| SegTrack | segtrack/drift/obj2 |  | 0.955 |  | +0.003 |  |  | +0.001 | A1z |
| SegTrack | segtrack/monkey/obj1 |  | 0.940 |  | +0.004 |  |  | +0.002 | A1z |
| SegTrack | segtrack/monkeydog/obj1 |  | 0.917 |  | -0.021 |  |  | -0.021 | A1z |
| SegTrack | segtrack/monkeydog/obj2 |  | 0.929 |  | -0.011 |  |  | +0.000 | B1 |
| SegTrack | segtrack/parachute/obj1 |  | 0.965 | +0.005 | +0.006 | +0.003 | +0.005 | +0.006 | A1z |
