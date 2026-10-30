"""
game/modes/pve.py
- 플레이어 vs AI 대전 모드 (PvE).
- 일반 모드(랜덤 셔플 선택) 및 엣지 실시간성을 활용한 무적 얌체 모드(God Mode) 지원.
"""

import random

from configs.app_config import GamePlayConfig

from game.modes.base import BaseGameMode
from game.rules import get_counter_gesture
from game.types import Detection, GameState, Gesture


class PvEMode(BaseGameMode):
    """플레이어 vs AI 대전 모드 (일반 / 무적 얌체 모드 지원)"""

    def __init__(self, config: GamePlayConfig, god_mode: bool = False) -> None:
        super().__init__(config)
        self.god_mode = god_mode
        self._pc_fixed_gesture: Gesture = Gesture.NONE

    def _on_round_start(self) -> None:
        """수동·자동 라운드 시작마다 일반 모드의 PC 패를 새로 선택."""
        if not self.god_mode:
            self._pc_fixed_gesture = random.choice(
                [Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS]
            )

    def extract_player_gestures(
        self,
        detections: list[Detection],
        frame_width: int,
    ) -> tuple[Gesture, Gesture]:
        p1_gesture = Gesture.NONE
        if detections:
            best_det = max(detections, key=lambda d: d.confidence)
            p1_gesture = best_det.gesture

        current_state = self.state_machine.context.state

        if current_state == GameState.COUNTDOWN:
            # 카운트다운 룰렛 슬롯머신 연출용 패
            pc_gesture = random.choice([Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS])
            # UI가 카운트다운 중에도 읽을 수 있도록 버퍼에 임시 기록
            self.state_machine.context.p2_vote_buffer.append(pc_gesture)
        elif current_state == GameState.JUDGING:
            if self.god_mode:
                pc_gesture = (
                    get_counter_gesture(p1_gesture)
                    if p1_gesture != Gesture.NONE
                    else Gesture.ROCK
                )
            else:
                pc_gesture = self._pc_fixed_gesture
        else:
            pc_gesture = self._pc_fixed_gesture

        return p1_gesture, pc_gesture
