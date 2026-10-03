from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\msilo\OneDrive\과제연구2")
OUT = ROOT / "sam2_tutorial_assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT_REG = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


INK = "#1F2937"
MUTED = "#667085"
BLUE = "#2563EB"
BLUE_BG = "#EAF2FF"
TEAL = "#0F766E"
TEAL_BG = "#E7F8F5"
ORANGE = "#EA580C"
ORANGE_BG = "#FFF1E8"
RED = "#DC2626"
RED_BG = "#FEECEC"
PURPLE = "#7C3AED"
PURPLE_BG = "#F2ECFF"
GRAY_BG = "#F5F7FA"
LINE = "#CBD5E1"


def canvas(title, subtitle):
    img = Image.new("RGB", (1800, 1050), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((35, 30, 1765, 1020), radius=28, fill="white", outline="#D7DEE8", width=3)
    d.text((80, 65), title, font=font(42, True), fill=INK)
    d.text((82, 125), subtitle, font=font(23), fill=MUTED)
    d.line((80, 175, 1720, 175), fill="#E4E9F0", width=3)
    return img, d


def box(d, xy, title, body, fill, outline, title_color=INK, body_color=MUTED, title_size=27, body_size=20):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=22, fill=fill, outline=outline, width=3)
    d.text((x1 + 24, y1 + 20), title, font=font(title_size, True), fill=title_color)
    lines = body.split("\n")
    y = y1 + 66
    for line in lines:
        d.text((x1 + 24, y), line, font=font(body_size), fill=body_color)
        y += body_size + 13


def arrow(d, start, end, color=INK, width=5):
    d.line((start, end), fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    import math
    a = math.atan2(y2-y1, x2-x1)
    for delta in (2.55, -2.55):
        p = (x2 + 20*math.cos(a+delta), y2 + 20*math.sin(a+delta))
        d.line((end, p), fill=color, width=width)


def cv_task_map():
    img, d = canvas("컴퓨터비전 과제 지도", "같은 사진을 보더라도 모델에게 묻는 질문이 달라지면 과제와 정답의 형태가 달라진다")
    box(d, (90, 220, 410, 400), "① 분류", "질문: 사진 전체가 무엇인가?\n출력: 고양이 92%", BLUE_BG, BLUE)
    box(d, (480, 220, 800, 400), "② 객체 탐지", "질문: 어디에 무엇이 있는가?\n출력: 상자 + 종류", TEAL_BG, TEAL)
    box(d, (870, 220, 1190, 400), "③ 분할", "질문: 어느 픽셀까지 객체인가?\n출력: 픽셀 단위 마스크", ORANGE_BG, ORANGE)
    box(d, (1260, 220, 1710, 400), "④ 영상 추적·VOS", "질문: 같은 객체가 다음 프레임에서 어디인가?\n출력: 시간에 따라 이어지는 마스크 + 정체성", PURPLE_BG, PURPLE)
    arrow(d, (410, 310), (475, 310), "#94A3B8", 4)
    arrow(d, (800, 310), (865, 310), "#94A3B8", 4)
    arrow(d, (1190, 310), (1255, 310), "#94A3B8", 4)
    d.text((90, 470), "분할의 세 종류", font=font(30, True), fill=INK)
    box(d, (90, 530, 580, 750), "Semantic segmentation", "같은 종류는 한 덩어리로 표시\n예: 사람 픽셀 전체 = 사람", GRAY_BG, "#94A3B8")
    box(d, (655, 530, 1145, 750), "Instance segmentation", "같은 종류라도 개체를 따로 표시\n예: 사람 1, 사람 2", GRAY_BG, "#94A3B8")
    box(d, (1220, 530, 1710, 750), "Panoptic segmentation", "배경 영역 + 개체 구분을 동시에\n예: 도로·하늘 + 사람별 마스크", GRAY_BG, "#94A3B8")
    d.rounded_rectangle((90, 820, 1710, 945), radius=18, fill="#F8FAFC", outline=LINE, width=2)
    d.text((120, 845), "SAM 2의 중심 과제", font=font(25, True), fill=PURPLE)
    d.text((370, 845), "사용자가 점·상자·마스크로 지정한 객체를 픽셀 단위로 나누고, 영상 전체에서 같은 객체로 이어 간다.", font=font(24), fill=INK)
    d.text((120, 890), "핵심 난점", font=font(25, True), fill=RED)
    d.text((280, 890), "가림·빠른 움직임·비슷한 방해물 때문에 과거의 틀린 마스크가 다음 프레임 오류로 증폭될 수 있다.", font=font(24), fill=INK)
    img.save(OUT / "01_cv_task_map.png")


def sam2_pipeline():
    img, d = canvas("SAM 2가 한 프레임을 처리하는 흐름", "현재 화면, 사용자 프롬프트, 과거 메모리를 합쳐 현재 객체 마스크를 만들고 다시 메모리에 기록한다")
    box(d, (90, 235, 360, 410), "현재 프레임", "RGB 픽셀\nH × W × 3", BLUE_BG, BLUE)
    box(d, (90, 510, 360, 685), "사용자 프롬프트", "점 · 상자 · 마스크\n‘이 객체를 따라가라’", ORANGE_BG, ORANGE)
    box(d, (455, 235, 765, 410), "Hiera 이미지 인코더", "픽셀 → 다중 해상도 특징\n모양·색·문맥을 숫자로 요약", BLUE_BG, BLUE)
    box(d, (455, 510, 765, 685), "프롬프트 인코더", "지시를 임베딩으로 변환\n공간적 힌트 생성", ORANGE_BG, ORANGE)
    box(d, (860, 210, 1200, 435), "메모리 어텐션", "현재 특징이 과거의 어떤 정보와\n관련 있는지 선택적으로 읽음\n현재 + 과거 특징을 결합", PURPLE_BG, PURPLE)
    box(d, (860, 515, 1200, 740), "스트리밍 메모리", "이전 마스크·특징·객체 포인터\n제한된 슬롯에 저장\n읽기·쓰기·삭제 정책이 중요", TEAL_BG, TEAL)
    box(d, (1295, 260, 1710, 500), "마스크 디코더", "결합된 특징 + 프롬프트로\n후보 마스크 여러 개 생성\n예측 IoU·object score도 출력", ORANGE_BG, ORANGE)
    box(d, (1295, 595, 1710, 805), "현재 결과", "선택된 객체 마스크\n품질·존재 점수\n다음 프레임의 입력이 됨", TEAL_BG, TEAL)
    arrow(d, (360, 320), (450, 320), BLUE)
    arrow(d, (360, 595), (450, 595), ORANGE)
    arrow(d, (765, 320), (855, 320), PURPLE)
    arrow(d, (765, 595), (955, 430), PURPLE)
    arrow(d, (1030, 515), (1030, 440), PURPLE)
    arrow(d, (1200, 325), (1290, 360), PURPLE)
    arrow(d, (1500, 500), (1500, 590), TEAL)
    arrow(d, (1295, 700), (1205, 650), TEAL)
    d.rounded_rectangle((90, 855, 1710, 975), radius=18, fill=RED_BG, outline="#F5A5A5", width=2)
    d.text((120, 882), "오류 누적의 출발점", font=font(25, True), fill=RED)
    d.text((375, 880), "틀린 현재 결과를 ‘믿을 만한 기억’으로 저장하면 다음 프레임이 그 오류를 참고하고,", font=font(21), fill=INK)
    d.text((375, 922), "그 오류가 다시 저장되는 자기증폭 고리가 생긴다.", font=font(21), fill=INK)
    img.save(OUT / "02_sam2_pipeline.png")


def drift_guard():
    img, d = canvas("같은 실수도 메모리 정책에 따라 결과가 달라진다", "무조건 저장하는 기본 흐름과, 위험을 보정해 선택적으로 저장·복구하는 흐름 비교")
    d.text((90, 215), "A. 무조건 메모리 갱신", font=font(30, True), fill=RED)
    xs = [110, 390, 670, 950, 1230, 1510]
    labels = ["정상 추적", "부분 가림", "틀린 마스크", "오류 저장", "드리프트", "다른 객체 추적"]
    fills = [TEAL_BG, ORANGE_BG, RED_BG, RED_BG, RED_BG, RED_BG]
    outlines = [TEAL, ORANGE, RED, RED, RED, RED]
    for i, x in enumerate(xs):
        d.rounded_rectangle((x, 270, x+190, 405), radius=18, fill=fills[i], outline=outlines[i], width=3)
        d.text((x+20, 300), f"t{i+1}", font=font(22, True), fill=outlines[i])
        d.text((x+20, 345), labels[i], font=font(20, True), fill=INK)
        if i < len(xs)-1: arrow(d, (x+190, 337), (xs[i+1]-10, 337), RED if i>=1 else "#94A3B8", 4)
    d.text((90, 485), "B. 위험 보정 + 선택적 메모리 + 앵커 복구", font=font(30, True), fill=TEAL)
    labels2 = ["정상 추적\nTRACK", "부분 가림\nSUSPECT", "저장 보류", "이전 앵커 유지", "재탐색", "정상 객체 복구\nTRACK"]
    fills2 = [TEAL_BG, ORANGE_BG, ORANGE_BG, BLUE_BG, PURPLE_BG, TEAL_BG]
    outlines2 = [TEAL, ORANGE, ORANGE, BLUE, PURPLE, TEAL]
    for i, x in enumerate(xs):
        d.rounded_rectangle((x, 550, x+190, 720), radius=18, fill=fills2[i], outline=outlines2[i], width=3)
        d.text((x+20, 575), f"t{i+1}", font=font(22, True), fill=outlines2[i])
        y=620
        for line in labels2[i].split("\n"):
            d.text((x+20, y), line, font=font(19, True), fill=INK); y+=31
        if i < len(xs)-1: arrow(d, (x+190, 635), (xs[i+1]-10, 635), TEAL if i>=1 else "#94A3B8", 4)
    d.rounded_rectangle((110, 800, 1700, 940), radius=20, fill="#F8FAFC", outline=LINE, width=2)
    d.text((140, 828), "연구의 핵심 가설", font=font(27, True), fill=PURPLE)
    d.text((390, 828), "‘점수가 낮다’만 보는 대신, 점수가 실제 미래 실패확률과 맞도록 보정하면 위험한 프레임을 더 잘 가려낼 수 있다.", font=font(23), fill=INK)
    d.text((140, 880), "검증해야 할 것", font=font(25, True), fill=BLUE)
    d.text((360, 880), "최종 J&F뿐 아니라 잘못된 메모리 기록률, 가림 후 회복 시간, 실패 검출 성능, 계산량까지 함께 좋아지는가?", font=font(23), fill=INK)
    img.save(OUT / "03_drift_vs_guard.png")


def calibration_controller():
    img, d = canvas("신뢰도 보정과 3상태 제어기", "모델의 원시 점수를 바로 믿지 않고 ‘실패 위험’으로 바꾼 뒤, 상태에 따라 다른 행동을 선택한다")
    box(d, (90, 240, 390, 465), "① 내부 신호", "예측 IoU\nobject score\n후보 간 차이\n마스크 변화량", BLUE_BG, BLUE)
    box(d, (485, 240, 810, 465), "② 보정기", "검증 데이터에서 학습\n원시 점수 → 실제 실패확률\nECE·Brier로 점검", PURPLE_BG, PURPLE)
    box(d, (905, 240, 1225, 465), "③ 위험 추정", "예: 다음 3프레임 안에\nIoU < 0.5가 될 확률\nq = 0.18", ORANGE_BG, ORANGE)
    arrow(d, (390, 350), (480, 350), PURPLE)
    arrow(d, (810, 350), (900, 350), PURPLE)
    d.text((1300, 215), "④ 상태와 행동", font=font(29, True), fill=INK)
    box(d, (1300, 265, 1710, 410), "TRACK — 위험 낮음", "마스크 사용 · 메모리 기록 허용", TEAL_BG, TEAL, title_size=24, body_size=20)
    box(d, (1300, 445, 1710, 590), "SUSPECT — 위험 중간", "저장 보류 · 후보 재검토 · 계산 추가", ORANGE_BG, ORANGE, title_size=24, body_size=20)
    box(d, (1300, 625, 1710, 770), "LOST — 위험 높음", "앵커/재탐색 사용 · 잘못된 정체성 차단", RED_BG, RED, title_size=24, body_size=20)
    arrow(d, (1225, 350), (1295, 335), TEAL)
    arrow(d, (1225, 365), (1295, 515), ORANGE)
    arrow(d, (1225, 380), (1295, 695), RED)
    d.rounded_rectangle((90, 550, 1225, 915), radius=22, fill="#F8FAFC", outline=LINE, width=2)
    d.text((125, 585), "보정 전과 보정 후의 차이", font=font(28, True), fill=INK)
    d.text((125, 650), "보정 전", font=font(24, True), fill=RED)
    d.text((270, 650), "신뢰도 0.90인 프레임 100개 중 실제 성공이 60개라면 숫자가 지나치게 낙관적이다.", font=font(22), fill=INK)
    d.text((125, 710), "보정 후", font=font(24, True), fill=TEAL)
    d.text((270, 710), "보정된 성공확률 0.60처럼 현실 빈도에 맞춘다. 값의 ‘순위’가 아니라 의미를 바로잡는다.", font=font(22), fill=INK)
    d.text((125, 785), "중요", font=font(24, True), fill=PURPLE)
    d.text((240, 785), "좋은 보정은 추적 정확도를 자동으로 올리는 마법이 아니다. 더 안전한 메모리·재탐색 결정을 가능하게 하는 측정 장치다.", font=font(22), fill=INK)
    d.text((125, 845), "연구 질문", font=font(24, True), fill=BLUE)
    d.text((280, 845), "한 데이터셋에서 배운 보정이 다른 데이터셋에서도 통하는가? 위험에 비례해 행동하면 성능–비용 균형이 좋아지는가?", font=font(22), fill=INK)
    img.save(OUT / "04_calibration_controller.png")


def metric_example():
    img, d = canvas("IoU와 Dice를 픽셀로 직접 계산하기", "정답 마스크와 예측 마스크의 겹침을 세어 분할 품질을 수치로 만든다")
    ox, oy, cell = 110, 245, 58
    gt = {(r,c) for r in range(2,7) for c in range(2,7)}
    pred = {(r,c) for r in range(3,8) for c in range(3,8)}
    for r in range(9):
        for c in range(9):
            x1, y1 = ox+c*cell, oy+r*cell
            ingt, inp = (r,c) in gt, (r,c) in pred
            if ingt and inp: fill="#7DD3FC"
            elif ingt: fill="#FDBA74"
            elif inp: fill="#C4B5FD"
            else: fill="white"
            d.rectangle((x1,y1,x1+cell,y1+cell),fill=fill,outline="#94A3B8",width=2)
    d.text((110, 790), "주황: 정답에만 있음(FN)", font=font(18), fill=ORANGE)
    d.text((335, 790), "보라: 예측에만 있음(FP)", font=font(18), fill=PURPLE)
    d.text((560, 790), "파랑: 둘이 겹침(TP)", font=font(18), fill=BLUE)
    inter=len(gt&pred); union=len(gt|pred)
    d.text((750, 245), f"정답 픽셀 수 |G| = {len(gt)}", font=font(27, True), fill=ORANGE)
    d.text((750, 300), f"예측 픽셀 수 |P| = {len(pred)}", font=font(27, True), fill=PURPLE)
    d.text((750, 355), f"교집합 |G ∩ P| = {inter}", font=font(27, True), fill=BLUE)
    d.text((750, 410), f"합집합 |G ∪ P| = {union}", font=font(27, True), fill=TEAL)
    d.rounded_rectangle((735, 485, 1660, 635),radius=20,fill=BLUE_BG,outline=BLUE,width=3)
    d.text((775, 520), f"IoU = 교집합 / 합집합 = {inter} / {union} = {inter/union:.3f}", font=font(31, True), fill=BLUE)
    d.rounded_rectangle((735, 675, 1660, 825),radius=20,fill=TEAL_BG,outline=TEAL,width=3)
    d.text((775, 710), f"Dice = 2×교집합 / (정답+예측) = {2*inter} / {len(gt)+len(pred)} = {2*inter/(len(gt)+len(pred)):.3f}", font=font(29, True), fill=TEAL)
    d.text((750, 875), "같은 예측에서 Dice가 IoU보다 크게 보이는 것은 공식의 스케일 차이다.", font=font(20), fill=MUTED)
    d.text((750, 915), "숫자가 더 크다고 해서 예측 자체가 더 좋아진 것은 아니다.", font=font(20), fill=MUTED)
    img.save(OUT / "05_iou_dice_example.png")


if __name__ == "__main__":
    cv_task_map()
    sam2_pipeline()
    drift_guard()
    calibration_controller()
    metric_example()
    print(len(list(OUT.glob("*.png"))))
