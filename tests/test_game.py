"""GPU·카메라 없이 실행하는 게임 규칙 및 상태 전이 회귀 검사."""

import unittest
from collections import deque
from unittest.mock import Mock, patch

from configs.app_config import GamePlayConfig
from game.modes.pve import PvEMode
from game.modes.pvp import PvPMode
from game.rules import get_counter_gesture, judge_round
from game.state_machine import GameStateMachine
from game.types import Detection, GameResult, GameState, Gesture


class GameTests(unittest.TestCase):
    def test_all_rule_combinations_and_missing_hands(self):
        gestures = [Gesture.PAPER, Gesture.ROCK, Gesture.SCISSORS]
        expected = [
            [GameResult.DRAW, GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN],
            [GameResult.PLAYER2_WIN, GameResult.DRAW, GameResult.PLAYER1_WIN],
            [GameResult.PLAYER1_WIN, GameResult.PLAYER2_WIN, GameResult.DRAW],
        ]
        for i, first in enumerate(gestures):
            for j, second in enumerate(gestures):
                with self.subTest(first=first, second=second):
                    self.assertEqual(judge_round(first, second), expected[i][j])
            self.assertEqual(judge_round(first, Gesture.NONE), GameResult.INVALID)
            self.assertEqual(judge_round(Gesture.NONE, first), GameResult.INVALID)
            self.assertEqual(
                judge_round(first, get_counter_gesture(first)), GameResult.PLAYER2_WIN
            )
        self.assertEqual(judge_round(Gesture.NONE, Gesture.NONE), GameResult.INVALID)

    def test_vote_missing_sparse_and_tied_observations(self):
        cases = [
            ([Gesture.NONE] * 10, Gesture.NONE),
            ([Gesture.NONE] * 9 + [Gesture.ROCK], Gesture.ROCK),
            ([Gesture.PAPER, Gesture.ROCK], Gesture.PAPER),
            ([Gesture.PAPER, Gesture.ROCK, Gesture.ROCK], Gesture.ROCK),
        ]
        for observations, expected in cases:
            with self.subTest(observations=observations):
                self.assertEqual(
                    GameStateMachine._get_majority_gesture(deque(observations)), expected
                )

    def test_pvp_split_and_confidence_selection(self):
        mode = PvPMode(GamePlayConfig())
        detections = [
            Detection(Gesture.PAPER, 0.7, (0, 0, 100, 100)),
            Detection(Gesture.ROCK, 0.9, (100, 0, 200, 100)),
            Detection(Gesture.SCISSORS, 0.8, (300, 0, 340, 100)),
        ]
        self.assertEqual(
            mode.extract_player_gestures(detections, 640),
            (Gesture.ROCK, Gesture.SCISSORS),
        )
        self.assertEqual(
            mode.extract_player_gestures([], 640), (Gesture.NONE, Gesture.NONE)
        )

    @patch('game.state_machine.time.time', return_value=100.0)
    def test_fsm_timing_vote_count_and_automatic_round(self, clock):
        on_start = Mock()
        fsm = GameStateMachine(GamePlayConfig(), on_round_start=on_start)
        fsm.start_round()
        clock.return_value = 102.9
        fsm.update(Gesture.ROCK, Gesture.SCISSORS)
        self.assertEqual(fsm.context.state, GameState.COUNTDOWN)
        clock.return_value = 103.0
        fsm.update(Gesture.ROCK, Gesture.SCISSORS)
        self.assertEqual(fsm.context.state, GameState.JUDGING)
        for _ in range(9):
            fsm.update(Gesture.ROCK, Gesture.SCISSORS)
        self.assertEqual(fsm.context.state, GameState.JUDGING)
        fsm.update(Gesture.ROCK, Gesture.SCISSORS)
        self.assertEqual(fsm.context.state, GameState.ROUND_RESULT)
        self.assertEqual((fsm.context.p1_score, fsm.context.round_count), (1, 1))
        clock.return_value = 105.4
        fsm.update(Gesture.NONE, Gesture.NONE)
        self.assertEqual(fsm.context.state, GameState.ROUND_RESULT)
        clock.return_value = 105.5
        fsm.update(Gesture.NONE, Gesture.NONE)
        self.assertEqual(fsm.context.state, GameState.COUNTDOWN)
        self.assertEqual(on_start.call_count, 2)

    def test_normal_pc_reselected_on_automatic_and_manual_rounds(self):
        config = GamePlayConfig(
            countdown_seconds=0, result_display_seconds=0, voting_window_frames=2
        )
        mode = PvEMode(config)
        hand = [Detection(Gesture.ROCK, 0.9, (0, 0, 10, 10))]
        with patch('game.modes.pve.random.choice', return_value=Gesture.ROCK):
            mode.start_round()
            for _ in range(3):
                mode.update(hand, 640)
        self.assertEqual(mode.context.p2_final_gesture, Gesture.ROCK)
        with patch('game.modes.pve.random.choice', return_value=Gesture.PAPER) as choice:
            mode.update(hand, 640)
            choice.assert_called_once()
            for _ in range(3):
                mode.update(hand, 640)
        self.assertEqual(mode.context.p2_final_gesture, Gesture.PAPER)
        with patch('game.modes.pve.random.choice', return_value=Gesture.SCISSORS):
            mode.start_round()
            for _ in range(3):
                mode.update(hand, 640)
        self.assertEqual(mode.context.p2_final_gesture, Gesture.SCISSORS)

    def test_target_score_cannot_be_bypassed_by_manual_start(self):
        for winner in ('p1_score', 'p2_score'):
            with self.subTest(winner=winner):
                on_start = Mock()
                fsm = GameStateMachine(GamePlayConfig(), on_round_start=on_start)
                fsm.context.state = GameState.ROUND_RESULT
                setattr(fsm.context, winner, 3)
                fsm.start_round()
                self.assertEqual(fsm.context.state, GameState.MATCH_OVER)
                self.assertEqual(getattr(fsm.context, winner), 3)
                on_start.assert_not_called()
                fsm.reset_match()
                fsm.start_round()
                self.assertEqual(fsm.context.state, GameState.COUNTDOWN)
                on_start.assert_called_once()

    def test_target_score_ends_match_automatically(self):
        fsm = GameStateMachine(GamePlayConfig(result_display_seconds=0))
        fsm.context.state = GameState.ROUND_RESULT
        fsm.context.p2_score = 3
        fsm.update(Gesture.NONE, Gesture.NONE)
        self.assertEqual(fsm.context.state, GameState.MATCH_OVER)

    def test_god_mode_selects_counter_for_best_detection(self):
        mode = PvEMode(GamePlayConfig(), god_mode=True)
        mode.context.state = GameState.JUDGING
        for gesture in (Gesture.PAPER, Gesture.ROCK, Gesture.SCISSORS):
            hand = [Detection(gesture, 0.9, (0, 0, 10, 10))]
            user, pc = mode.extract_player_gestures(hand, 640)
            self.assertEqual(judge_round(user, pc), GameResult.PLAYER2_WIN)

    def test_invalid_round_has_no_score_and_reset_clears_context(self):
        mode = PvPMode(GamePlayConfig(countdown_seconds=0, voting_window_frames=1))
        mode.start_round()
        mode.update([], 640)
        mode.update([], 640)
        self.assertEqual(mode.context.last_result, GameResult.INVALID)
        self.assertEqual((mode.context.p1_score, mode.context.p2_score), (0, 0))
        self.assertEqual(mode.context.round_count, 1)
        mode.reset_match()
        self.assertEqual(mode.context.state, GameState.IDLE)
        self.assertEqual(mode.context.round_count, 0)
        self.assertFalse(mode.context.p1_vote_buffer)
        self.assertFalse(mode.context.p2_vote_buffer)


if __name__ == '__main__':
    unittest.main()
