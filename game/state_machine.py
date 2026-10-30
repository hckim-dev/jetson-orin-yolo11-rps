"""
game/state_machine.py
- 게임 수명 주기(IDLE -> COUNTDOWN -> JUDGING -> ROUND_RESULT -> MATCH_OVER) 상태 머신.
- 판정 구간의 여러 프레임에서 유효 제스처의 최빈값을 선택.
"""

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

from configs.app_config import GamePlayConfig

from game.rules import judge_round
from game.types import GameResult, GameState, Gesture


@dataclass(slots=True)
class GameContext:
    """게임 세션의 실시간 상태 데이터 컨테이너"""

    state: GameState = GameState.IDLE
    state_timer: float = 0.0

    p1_score: int = 0
    p2_score: int = 0
    round_count: int = 0

    p1_final_gesture: Gesture = Gesture.NONE
    p2_final_gesture: Gesture = Gesture.NONE
    last_result: GameResult = GameResult.INVALID

    # 판정 구간의 프레임 수집 버퍼
    p1_vote_buffer: deque[Gesture] = field(default_factory=deque)
    p2_vote_buffer: deque[Gesture] = field(default_factory=deque)


class GameStateMachine:
    """가위바위보 상태 전이 및 윈도우 다수결 판정 관리 클래스"""

    def __init__(
        self, config: GamePlayConfig, on_round_start: Callable[[], None] | None = None
    ) -> None:
        self.config = config
        self._on_round_start = on_round_start
        self.context = GameContext(
            p1_vote_buffer=deque(maxlen=config.voting_window_frames),
            p2_vote_buffer=deque(maxlen=config.voting_window_frames),
        )

    def reset_match(self) -> None:
        """전체 매치(스코어 및 라운드) 완전 초기화"""
        self.context.state = GameState.IDLE
        self.context.state_timer = 0.0
        self.context.p1_score = 0
        self.context.p2_score = 0
        self.context.round_count = 0
        self.context.p1_final_gesture = Gesture.NONE
        self.context.p2_final_gesture = Gesture.NONE
        self.context.last_result = GameResult.INVALID
        self.context.p1_vote_buffer.clear()
        self.context.p2_vote_buffer.clear()

    def start_round(self) -> None:
        """새 라운드(카운트다운) 진입"""
        if (
            self.context.p1_score >= self.config.target_score
            or self.context.p2_score >= self.config.target_score
        ):
            self.context.state = GameState.MATCH_OVER
            return

        self.context.state = GameState.COUNTDOWN
        self.context.state_timer = time.time()
        self.context.p1_final_gesture = Gesture.NONE
        self.context.p2_final_gesture = Gesture.NONE
        self.context.last_result = GameResult.INVALID
        self.context.p1_vote_buffer.clear()
        self.context.p2_vote_buffer.clear()

        if self._on_round_start is not None:
            self._on_round_start()

    @staticmethod
    def _get_majority_gesture(buffer: deque[Gesture]) -> Gesture:
        """버퍼 내 유효한 제스처 중 최빈값(Mode) 추출"""
        valid_gestures = [g for g in buffer if g != Gesture.NONE]
        if not valid_gestures:
            return Gesture.NONE

        counter = Counter(valid_gestures)
        most_common_gesture, _ = counter.most_common(1)[0]
        return most_common_gesture

    def update(
        self,
        p1_gesture: Gesture,
        p2_gesture: Gesture,
    ) -> GameContext:
        """프레임 단위 상태 업데이트 및 상태 전이(State Transition)"""
        now = time.time()
        elapsed = now - self.context.state_timer

        match self.context.state:
            case GameState.IDLE:
                # 대기 상태: 키 입력(Space 등) 대기
                pass

            case GameState.COUNTDOWN:
                if elapsed >= self.config.countdown_seconds:
                    self.context.state = GameState.JUDGING
                    self.context.state_timer = now
                    self.context.p1_vote_buffer.clear()
                    self.context.p2_vote_buffer.clear()

            case GameState.JUDGING:
                # 지정된 프레임 수(예: 10프레임) 동안 감지 결과 축적
                self.context.p1_vote_buffer.append(p1_gesture)
                self.context.p2_vote_buffer.append(p2_gesture)

                if len(self.context.p1_vote_buffer) >= self.config.voting_window_frames:
                    p1_final = self._get_majority_gesture(self.context.p1_vote_buffer)
                    p2_final = self._get_majority_gesture(self.context.p2_vote_buffer)

                    self.context.p1_final_gesture = p1_final
                    self.context.p2_final_gesture = p2_final
                    self.context.last_result = judge_round(p1_final, p2_final)

                    # 스코어 갱신
                    if self.context.last_result == GameResult.PLAYER1_WIN:
                        self.context.p1_score += 1
                    elif self.context.last_result == GameResult.PLAYER2_WIN:
                        self.context.p2_score += 1

                    self.context.round_count += 1
                    self.context.state = GameState.ROUND_RESULT
                    self.context.state_timer = now

            case GameState.ROUND_RESULT:
                if elapsed >= self.config.result_display_seconds:
                    # 세트 목표 점수 달성 확인
                    if (
                        self.context.p1_score >= self.config.target_score
                        or self.context.p2_score >= self.config.target_score
                    ):
                        self.context.state = GameState.MATCH_OVER
                    else:
                        self.start_round()

            case GameState.MATCH_OVER:
                # 세트 종료 상태: 리셋 키 입력 대기
                pass

        return self.context
