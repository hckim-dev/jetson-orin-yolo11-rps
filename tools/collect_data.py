"""
tools/collect_data.py
- Jetson 엣지 환경 맞춤형 커스텀 데이터셋(가위/바위/보/배경) 실시간 연사 수집 도구.
- 라벨 접두어가 붙은 JPEG 이미지를 단일 디렉터리에 저장하는 유틸리티.
"""

import os
import time

import cv2

SAVE_DIR = "custom_rps_data"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

current_label = None
last_save_time = 0.0
save_interval = 0.25  # 0.25초마다 1장씩 연사
count = 0

print("📸 데이터 수집 모드 시작")
print("  - [R]: 바위(Rock) 연사 수집")
print("  - [S]: 가위(Scissors) 연사 수집")
print("  - [P]: 보(Paper) 연사 수집")
print("  - [B]: 손 없는 배경(Background) 연사 수집")
print("  - [Space]: 수집 일시정지 | [Q]: 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if current_label and (now - last_save_time >= save_interval):
        count += 1
        img_name = f"{current_label}_{int(now * 1000)}.jpg"
        cv2.imwrite(os.path.join(SAVE_DIR, img_name), frame)
        last_save_time = now

    # 화면에 가이드 텍스트 표시
    status_text = f"Recording: {current_label}" if current_label else "IDLE"
    cv2.putText(
        frame,
        f"{status_text} (Total: {count})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if current_label else (0, 255, 255),
        2,
    )
    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord("r"):
        current_label = "rock"
    elif key == ord("s"):
        current_label = "scissors"
    elif key == ord("p"):
        current_label = "paper"
    elif key == ord("b"):
        current_label = "bg"
    elif key == ord(" "):
        current_label = None

cap.release()
cv2.destroyAllWindows()
