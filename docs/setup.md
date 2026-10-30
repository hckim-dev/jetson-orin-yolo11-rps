# 환경·모델 준비와 실행

[README로 돌아가기](../README.md)

## 실행 조건

대상은 Jetson의 Linux 카메라·CUDA 환경입니다. 아래는 소스에서 확인한 요구 조건이며, 특정 JetPack 조합의 실기 검증 목록은 아닙니다.

| 항목 | 요구 조건 |
|---|---|
| Python | 3.10 이상 문법 사용: `match`, `dataclass(slots=True)`, union 타입 |
| TensorRT | `execute_async_v3`, `get_tensor_name`, `set_tensor_address`를 사용하는 10.x 대상 코드 |
| CUDA / PyCUDA | `pycuda.autoinit`, Device 메모리, Pinned Host 메모리와 Stream 지원 |
| NumPy | 텐서 변환과 후처리에 사용 |
| OpenCV | 카메라 입력, `dnn.NMSBoxes`, GUI 창 지원. headless 패키지만으로는 화면 출력 불가 |
| 카메라·디스플레이 | 기본 장치 0, V4L2 우선 연결, OpenCV 창을 표시할 데스크톱 세션 |

JetPack·CUDA·TensorRT·Python 조합의 검증 버전은 기록되어 있지 않습니다. 먼저 애플리케이션을 실행할 Python 환경에서 필요한 모듈과 버전을 확인하세요.

```bash
python3 --version
python3 -c "import cv2, numpy, tensorrt, pycuda.driver; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__); print('TensorRT', tensorrt.__version__)"
```

패키지 import 성공만으로 GPU 엔진·카메라·GUI 동작까지 확인되지는 않습니다.

## 모델 엔진 준비

기본 경로는 프로젝트 루트 기준 `models/rps_yolo11n_custom_640.engine`입니다. 디렉터리를 만들고 실행 환경에서 사용할 호환 엔진을 배치하세요. 경로는 [VisionConfig](../configs/app_config.py)에서 변경할 수 있습니다.

모델 가중치·ONNX·TensorRT 엔진과 학습 데이터는 저장소에 포함되어 있지 않습니다. 호환 엔진을 별도로 준비해야 하며, 학습·export·엔진 빌드 스크립트와 모델 다운로드 링크는 제공하지 않습니다.

현재 래퍼와 후처리가 기대하는 모델 조건은 다음과 같습니다.

| 항목 | 코드의 가정 |
|---|---|
| 입력·출력 | 엔진의 텐서 인덱스 0이 입력, 1이 출력인 단일 입력·출력 구조 |
| 입력 shape | 정적 `(1, 3, 640, 640)`, RGB, NCHW, 0~1 정규화 |
| 입력 dtype | 전처리는 float32 생성. `np.copyto`로 엔진 dtype의 Host 버퍼에 복사 |
| 출력 shape | 정적 `(1, 7, N)`. 코드 주석의 640 입력 모델 예시는 `N=8400` |
| 출력 내용 | `[cx, cy, w, h, paper_score, rock_score, scissors_score]`, 좌표는 Letterbox 입력 픽셀 기준 |
| 클래스 순서 | `0: paper`, `1: rock`, `2: scissors` |
| NMS | 모델 외부의 OpenCV 후처리에서 수행 |

`N`은 검출 후보 수이며 실제 엔진의 출력 shape에서 가져옵니다. 동적 shape 설정, 여러 출력, NMS가 포함된 다른 출력 형식은 현재 래퍼에서 처리하지 않습니다.

입력 복사 시 엔진 dtype에 따라 변환될 수 있으므로 전처리 dtype과 엔진 입력 dtype을 구분해야 합니다. 예를 들어 NumPy의 float32→float16 복사는 가능하지만, 실제 엔진의 입출력 조건과 대상 장치에서의 실행은 별도로 확인해야 합니다. 엔진 내부 연산의 FP16 사용 여부도 파일 이름이나 입력 dtype만으로 확인할 수 없습니다.

## 실행과 설정

프로젝트 루트에서 실행합니다.

```bash
python3 main.py
```

[app_config.py](../configs/app_config.py)의 주요 기본값입니다.

| 설정 | 기본값 |
|---|---|
| 카메라 장치·요청 해상도 | `0`, `640×480` |
| 캡처 포맷·버퍼 | MJPG 요청, 버퍼 크기 `1` 요청 |
| 모델 입력 크기 | `640×640` |
| Confidence / IoU threshold | `0.50` / `0.45` |
| 카운트다운 / 결과 표시 | `3.0초` / `2.5초` |
| 판정 프레임 수 / 목표 점수 | `10` / `3승` |

카메라 속성 요청의 성공 여부와 실제 적용값을 현재 코드에서 검사하지 않습니다. `CameraConfig.fps=30`도 현재 캡처 설정에 전달되지 않아 FPS 제한이나 실제 촬영 속도를 뜻하지 않습니다.

최초 모드는 PvP입니다. SPACE로 시작하고 `1`·`2`·`3`으로 모드 변경, `r`로 초기화, `q` 또는 ESC로 종료합니다. 영문 키는 소문자 입력 기준입니다. 상세 동작은 [README](../README.md)의 조작 표를 참고하세요.

## 데이터 수집 도구

```bash
python3 tools/collect_data.py
```

`r`·`s`·`p`·`b`로 바위·가위·보·배경 수집을 선택하고 SPACE로 일시정지합니다. `q` 또는 ESC로 종료합니다. 현재 작업 디렉터리의 `custom_rps_data/`에 라벨 접두어가 붙은 JPEG를 0.25초 간격으로 저장합니다. 실제 속도는 캡처·저장 처리 시간에 영향을 받습니다.

폴더별 분류나 YOLO 바운딩 박스 주석 생성은 수행하지 않습니다. 배경 이미지는 수집용 구분이며 추론 모델의 네 번째 클래스가 아닙니다. 수집 데이터와 모델 파일은 `.gitignore`로 제외됩니다.

## 실행 실패 시 확인

| 증상 | 확인할 내용 |
|---|---|
| `ModuleNotFoundError` | 현재 Python 환경에 필요한 모듈이 설치되어 있는지 확인 |
| 엔진 파일 없음 | 기본 엔진 경로 또는 `VisionConfig.engine_path` 확인 |
| 엔진 로드·추론 실패 | 대상 장치의 런타임과 엔진, 정적 shape·dtype·클래스·출력 형식 확인 |
| 카메라 열기·프레임 수신 실패 | 장치 번호, 접근 권한, 다른 프로세스의 사용 여부, 지원 캡처 모드 확인 |
| 창 표시 실패 | GUI 지원 OpenCV와 디스플레이 세션 확인 |
