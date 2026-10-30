# TensorRPS

**Jetson에서 카메라 영상의 손 제스처를 인식해 진행하는 가위바위보 애플리케이션입니다.**

YOLO11 제스처 탐지 모델을 TensorRT로 실행하고, 탐지 결과를 게임 상태 머신과 OpenCV HUD에 연결했습니다. 카메라 한 대로 2인 대전과 PC 대전을 지원합니다.

**주요 기술**: Python · TensorRT 10.x · CUDA/PyCUDA · NumPy · OpenCV

## 시연 영상

https://github.com/user-attachments/assets/56a1ff8a-05af-4022-92a5-0b77f87816be

카운트다운, 제스처 판정, 점수 표시와 PvP·PvE 모드 전환 시연입니다.

## 주요 기능

| 기능 | 동작 |
|---|---|
| PvP | 화면 좌우에서 각각 신뢰도가 가장 높은 손을 선택해 2인 대전 |
| PvE Normal | 라운드마다 PC의 패를 무작위로 선택 |
| PvE God Mode | 판정 중 인식한 사용자 제스처에 대해 이기는 패를 선택 |
| 다중 프레임 판정 | 10프레임을 모아 유효 제스처의 최빈값으로 최종 패 결정 |
| 매치 진행 | 3초 카운트다운, 결과·점수 표시, 다음 라운드 자동 시작, 먼저 3승하면 종료 |

## 핵심 구현

- **반복 추론의 메모리 관리**: Pinned Host·Device 입출력 버퍼를 초기화 시 할당하고 재사용합니다. 같은 CUDA Stream에서 전송·추론·결과 복사를 수행합니다. [구현](core/trt_engine.py)
- **모드 간 게임 진행 공통화**: 플레이어 제스처 선택은 모드별 전략으로 분리하고, 라운드·점수·매치 종료는 공통 FSM에서 관리합니다. [구현](game/modes/base.py)
- **판정 구간의 관측 집계**: 카운트다운 이후 10프레임을 수집해 유효 제스처의 최빈값을 선택합니다. 미검출과 동률 처리 기준을 명시했습니다. [구현](game/state_machine.py)
- **화면 오버레이의 복사 범위 제한**: HUD 블렌딩을 오버레이 ROI에 적용해 전체 프레임 복사를 피합니다. 검출 박스·제스처 아이콘·점수를 같은 화면에 표시합니다. [구현](ui/renderer.py)

## 실행

Jetson의 CUDA·TensorRT 환경, GUI를 지원하는 OpenCV, NumPy, PyCUDA와 호환 모델 엔진이 필요합니다. **모델 가중치와 엔진은 저장소에 포함되어 있지 않습니다.** 모델 조건과 환경 확인 절차는 [실행 가이드](docs/setup.md)를 참고하세요.

프로젝트 루트에서 엔진을 `models/rps_yolo11n_custom_640.engine`에 준비한 뒤 실행합니다.

```bash
python3 main.py
```

| 키 | 기능 |
|---|---|
| `SPACE` | 라운드 시작 / 목표 점수 미달 시 결과 화면에서 다음 라운드로 진행 |
| `1` / `2` / `3` | PvP / PvE Normal / PvE God Mode로 전환하고 해당 매치 초기화 |
| `r` | 현재 매치 초기화 |
| `q` / `ESC` | 종료 |

## 구조와 상세 문서

```text
main.py                 카메라·탐지·게임·UI를 연결하는 메인 루프
configs/                비전·카메라·게임 설정
core/                   전처리·TensorRT 추론·후처리
game/                   규칙·상태 머신·모드별 제스처 선택
ui/                     OpenCV HUD 렌더링
tools/                  학습용 이미지 수집 도구
tests/                  하드웨어 독립 게임 로직 테스트
docs/                   상세 문서와 UI 아이콘(docs/assets/)
```

- [아키텍처와 설계 판단](docs/architecture.md)
- [환경·모델 준비와 실행](docs/setup.md)
- [검증 결과와 한계](docs/validation.md)

게임 로직은 다음 명령으로 별도 GPU 의존성 없이 확인할 수 있습니다.

```bash
python3 -B -m unittest discover -s tests -v
```
