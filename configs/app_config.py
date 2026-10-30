"""
configs/app_config.py
- 프로젝트 전역의 하이퍼파라미터 및 설정을 중앙 관리하는 모듈.
- Vision(TensorRT), Camera(V4L2), GamePlay 설정을 불변(frozen=True) 데이터클래스로 제공.
"""

from dataclasses import dataclass, field
from pathlib import Path

# 프로젝트 루트 디렉터리 절대 경로 확정
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """비전 엔진 및 TensorRT 관련 설정"""

    engine_path: Path = _PROJECT_ROOT / "models" / "rps_yolo11n_custom_640.engine"
    input_size: tuple[int, int] = (640, 640)
    conf_threshold: float = 0.50
    iou_threshold: float = 0.45


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """카메라 디바이스 및 비디오 스트림 설정"""

    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass(frozen=True, slots=True)
class GamePlayConfig:
    """게임 상태 머신 및 룰 설정"""

    countdown_seconds: float = 3.0
    result_display_seconds: float = 2.5
    voting_window_frames: int = 10  # 최빈값 판정 시 수집할 프레임 수
    target_score: int = 3  # 먼저 3승한 플레이어 승리 (무승부·무효 라운드 포함 횟수 제한 없음)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """전체 애플리케이션 통합 설정"""

    vision: VisionConfig = field(default_factory=VisionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    game: GamePlayConfig = field(default_factory=GamePlayConfig)
