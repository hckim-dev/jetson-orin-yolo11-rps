"""
ui/renderer.py
- 고품질 사이버 HUD, 스코어보드 뱃지, 브래킷 카드 및 오버레이 렌더러
- ROI 인플레이스 알파 블렌딩 및 안전 클램핑 적용
"""

import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from game.state_machine import GameContext
from game.types import Detection, GameMode, GameResult, GameState, Gesture

from ui.colors import (
    COLOR_AMBER,
    COLOR_BG_DARK,
    COLOR_BLUE,
    COLOR_BORDER,
    COLOR_CARD_BG,
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    GESTURE_COLORS,
)


class GameUIRenderer:
    """게임 HUD, 바운딩 박스 및 그래픽 오버레이 렌더러"""

    FONT_MAIN: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_HEAD: int = cv2.FONT_HERSHEY_DUPLEX

    _ICON_FILES: ClassVar[dict[Gesture, str]] = {
        Gesture.ROCK: "rock.png",
        Gesture.PAPER: "paper.png",
        Gesture.SCISSORS: "scissors.png",
    }

    def __init__(
        self, countdown_sec: float = 3.0, assets_dir: Path | str | None = None
    ) -> None:
        self.countdown_sec = countdown_sec
        self.assets_dir = (
            Path(assets_dir)
            if assets_dir
            else Path(__file__).resolve().parent.parent / "docs" / "assets"
        )
        self.icons: dict[Gesture, np.ndarray | None] = self._load_icons()

    def _load_icons(self) -> dict[Gesture, np.ndarray | None]:
        """assets 폴더에서 가위바위보 아이콘 PNG 로드"""
        loaded: dict[Gesture, np.ndarray | None] = {}
        for gesture, filename in self._ICON_FILES.items():
            path = self.assets_dir / filename
            loaded[gesture] = (
                cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
            )
        loaded[Gesture.NONE] = None
        return loaded

    # --- 최적화된 렌더링 헬퍼 ---
    @staticmethod
    def _draw_centered_text(
        frame: np.ndarray,
        text: str,
        center_x: int,
        y: int,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 1.0,
        color: tuple[int, int, int] = COLOR_WHITE,
        thickness: int = 2,
    ) -> None:
        (tw, _), _ = cv2.getTextSize(text, font_face, scale, thickness)
        cv2.putText(
            frame,
            text,
            (center_x - tw // 2, y),
            font_face,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_overlay_rect(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        alpha: float = 0.75,
    ) -> None:
        """ROI 영역만 추출하여 인플레이스 블렌딩 (전체 복사 메모리 오버헤드 제거)"""
        h, w = frame.shape[:2]
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w, x2), min(h, y2)

        if x1_c >= x2_c or y1_c >= y2_c:
            return

        sub_roi = frame[y1_c:y2_c, x1_c:x2_c]
        color_rect = np.full_like(sub_roi, color, dtype=np.uint8)
        cv2.addWeighted(color_rect, alpha, sub_roi, 1.0 - alpha, 0, sub_roi)

    @staticmethod
    def _draw_corner_brackets(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        length: int = 12,
        thickness: int = 2,
    ) -> None:
        """HUD 스타일 모서리 브래킷"""
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)

    def _overlay_icon(
        self,
        frame: np.ndarray,
        gesture: Gesture,
        center_x: int,
        center_y: int,
        size: int = 70,
    ) -> None:
        half_sz = size // 2
        x = center_x - half_sz
        y = center_y - half_sz
        h, w = frame.shape[:2]

        if x < 0 or y < 0 or x + size > w or y + size > h:
            return

        icon = self.icons.get(gesture)
        if icon is not None:
            resized = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    frame[y : y + size, x : x + size, c] = (
                        alpha * resized[:, :, c]
                        + (1.0 - alpha) * frame[y : y + size, x : x + size, c]
                    )
                return
            frame[y : y + size, x : x + size] = resized[:, :, :3]
            return

        color = GESTURE_COLORS.get(gesture, COLOR_GRAY)
        self._draw_overlay_rect(
            frame, x, y, x + size, y + size, COLOR_CARD_BG, alpha=0.8
        )
        self._draw_corner_brackets(
            frame, x, y, x + size, y + size, color, length=8, thickness=1
        )
        label = "?" if gesture == Gesture.NONE else gesture.name[:3]
        self._draw_centered_text(
            frame,
            label,
            center_x,
            center_y + 7,
            font_face=self.FONT_HEAD,
            scale=0.75,
            color=color,
            thickness=2,
        )

    # --- 비전 검출 결과 렌더링 ---
    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """화면 상단 이탈 방지 클램핑 적용 바운딩 박스"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = GESTURE_COLORS.get(det.gesture, COLOR_WHITE)

            self._draw_overlay_rect(frame, x1, y1, x2, y2, color, alpha=0.15)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            self._draw_corner_brackets(
                frame, x1, y1, x2, y2, color, length=14, thickness=2
            )

            label = f"{det.gesture.name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, self.FONT_MAIN, 0.45, 1)

            # 상단 경계선 안전 클램핑 (y1이 너무 낮으면 박스 안쪽으로 렌더링)
            label_top = y1 - th - 8 if y1 - th - 8 > 0 else y1 + 2
            label_bottom = y1 if y1 - th - 8 > 0 else y1 + th + 8
            text_y = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 4

            cv2.rectangle(
                frame, (x1, label_top), (x1 + tw + 8, label_bottom), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 4, text_y),
                self.FONT_MAIN,
                0.45,
                COLOR_BG_DARK,
                1,
                cv2.LINE_AA,
            )

    # --- 메인 상단 HUD ---
    def draw_hud(
        self,
        frame: np.ndarray,
        context: GameContext,
        mode: GameMode,
        fps: float,
    ) -> None:
        h, w = frame.shape[:2]

        # 1. 상단 헤더 바
        self._draw_overlay_rect(frame, 0, 0, w, 52, COLOR_BG_DARK, alpha=0.85)
        cv2.line(frame, (0, 52), (w, 52), COLOR_BORDER, 1, cv2.LINE_AA)
        cv2.line(frame, (0, 52), (180, 52), COLOR_AMBER, 2, cv2.LINE_AA)

        # 2. 좌측 모드 & 성능 인디케이터
        cv2.circle(frame, (16, 20), 4, COLOR_GREEN, -1)
        cv2.putText(
            frame,
            f"MODE: {mode.display_name}",
            (26, 24),
            self.FONT_MAIN,
            0.42,
            COLOR_AMBER,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | JETSON ORIN",
            (26, 42),
            self.FONT_MAIN,
            0.38,
            COLOR_GRAY,
            1,
            cv2.LINE_AA,
        )

        # 3. 중앙 뱃지형 스코어보드
        center_x = w // 2
        p1_label = "PLAYER 1"
        p2_label = "PLAYER 2" if mode.is_pvp else "PC"

        pill_w = 90
        self._draw_overlay_rect(
            frame,
            center_x - pill_w // 2,
            8,
            center_x + pill_w // 2,
            44,
            COLOR_CARD_BG,
            alpha=0.9,
        )
        cv2.rectangle(
            frame,
            (center_x - pill_w // 2, 8),
            (center_x + pill_w // 2, 44),
            COLOR_BORDER,
            1,
        )

        score_num = f"{context.p1_score}  :  {context.p2_score}"
        self._draw_centered_text(
            frame,
            score_num,
            center_x,
            32,
            font_face=self.FONT_HEAD,
            scale=0.65,
            color=COLOR_WHITE,
            thickness=2,
        )

        (p1_w, _), _ = cv2.getTextSize(p1_label, self.FONT_MAIN, 0.5, 1)
        cv2.putText(
            frame,
            p1_label,
            (center_x - pill_w // 2 - p1_w - 12, 31),
            self.FONT_MAIN,
            0.5,
            COLOR_CYAN,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            p2_label,
            (center_x + pill_w // 2 + 12, 31),
            self.FONT_MAIN,
            0.5,
            COLOR_AMBER,
            2,
            cv2.LINE_AA,
        )

        # 4. PvP 분할선 (바닥 끝까지 연결) 또는 PvE PC 선택 카드
        if mode.is_pvp:
            for y_pos in range(54, h, 16):
                cv2.line(
                    frame,
                    (w // 2, y_pos),
                    (w // 2, min(y_pos + 8, h)),
                    COLOR_BORDER,
                    1,
                    cv2.LINE_AA,
                )
        else:
            self._draw_pve_pc_card(frame, context)

    def _draw_pve_pc_card(self, frame: np.ndarray, context: GameContext) -> None:
        """우측 상단 PC 선택 카드"""
        w = frame.shape[1]
        card_w, card_h = 120, 115
        card_x, card_y = w - card_w - 15, 65

        self._draw_overlay_rect(
            frame,
            card_x,
            card_y,
            card_x + card_w,
            card_y + card_h,
            COLOR_BG_DARK,
            alpha=0.75,
        )
        cv2.rectangle(
            frame, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_BORDER, 1
        )
        self._draw_corner_brackets(
            frame,
            card_x,
            card_y,
            card_x + card_w,
            card_y + card_h,
            COLOR_AMBER,
            length=10,
            thickness=2,
        )

        card_center_x = card_x + card_w // 2
        self._draw_centered_text(
            frame,
            "PC CHOICE",
            center_x=card_center_x,
            y=card_y + 20,
            font_face=self.FONT_MAIN,
            scale=0.42,
            color=COLOR_AMBER,
            thickness=1,
        )

        display_gesture = Gesture.NONE
        if context.state == GameState.ROUND_RESULT:
            display_gesture = context.p2_final_gesture
        elif (
            context.state in (GameState.COUNTDOWN, GameState.JUDGING)
            and context.p2_vote_buffer
        ):
            display_gesture = context.p2_vote_buffer[-1]

        content_center_y = card_y + 25 + (card_h - 25) // 2
        if display_gesture != Gesture.NONE:
            self._overlay_icon(
                frame,
                display_gesture,
                center_x=card_center_x,
                center_y=content_center_y,
                size=62,
            )
        else:
            self._draw_centered_text(
                frame,
                "???",
                center_x=card_center_x,
                y=content_center_y + 10,
                font_face=self.FONT_HEAD,
                scale=0.9,
                color=COLOR_GRAY,
                thickness=2,
            )

    # --- 상태별 중앙 배너 및 인터랙션 렌더링 ---
    def draw_state_overlays(
        self, frame: np.ndarray, context: GameContext, mode: GameMode
    ) -> None:
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        match context.state:
            case GameState.IDLE:
                # 하단 풀 와이드 티커 바 (28px 높이로 바닥에 밀착)
                bar_h = 28
                bar_y1 = h - bar_h
                self._draw_overlay_rect(
                    frame, 0, bar_y1, w, h, COLOR_BG_DARK, alpha=0.88
                )
                cv2.line(frame, (0, bar_y1), (w, bar_y1), COLOR_BORDER, 1, cv2.LINE_AA)

                # 좌측: 플레이 조작 안내
                cv2.putText(
                    frame,
                    "[SPACE] START",
                    (16, bar_y1 + 19),
                    self.FONT_HEAD,
                    0.42,
                    COLOR_GREEN,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "[R] RESET",
                    (145, bar_y1 + 19),
                    self.FONT_HEAD,
                    0.42,
                    COLOR_GRAY,
                    1,
                    cv2.LINE_AA,
                )

                # 우측: 모드 변경 안내 (우측 정렬)
                mode_guide = "MODE :  [1] PvP   [2] PvE   [3] God"
                (tw, _), _ = cv2.getTextSize(mode_guide, self.FONT_MAIN, 0.40, 1)
                cv2.putText(
                    frame,
                    mode_guide,
                    (w - tw - 16, bar_y1 + 19),
                    self.FONT_MAIN,
                    0.40,
                    COLOR_AMBER,
                    1,
                    cv2.LINE_AA,
                )

            case GameState.COUNTDOWN:
                elapsed = time.time() - context.state_timer
                remain = max(0.0, self.countdown_sec - elapsed)
                count_str = str(int(np.ceil(remain))) if remain > 0.5 else "READY!"
                color = COLOR_AMBER if remain > 0.5 else COLOR_GREEN

                self._draw_overlay_rect(
                    frame,
                    center_x - 80,
                    center_y - 70,
                    center_x + 80,
                    center_y + 70,
                    COLOR_BG_DARK,
                    alpha=0.65,
                )
                self._draw_corner_brackets(
                    frame,
                    center_x - 80,
                    center_y - 70,
                    center_x + 80,
                    center_y + 70,
                    color,
                    length=15,
                    thickness=2,
                )
                self._draw_centered_text(
                    frame,
                    count_str,
                    center_x,
                    center_y + 25,
                    font_face=self.FONT_HEAD,
                    scale=2.2,
                    color=color,
                    thickness=4,
                )

            case GameState.JUDGING:
                self._draw_overlay_rect(
                    frame,
                    center_x - 120,
                    center_y - 35,
                    center_x + 120,
                    center_y + 35,
                    COLOR_BG_DARK,
                    alpha=0.75,
                )
                self._draw_corner_brackets(
                    frame,
                    center_x - 120,
                    center_y - 35,
                    center_x + 120,
                    center_y + 35,
                    COLOR_CYAN,
                    length=12,
                    thickness=2,
                )
                self._draw_centered_text(
                    frame,
                    "JUDGING...",
                    center_x,
                    center_y + 11,
                    font_face=self.FONT_HEAD,
                    scale=1.0,
                    color=COLOR_CYAN,
                    thickness=2,
                )

            case GameState.ROUND_RESULT:
                if context.last_result == GameResult.PLAYER1_WIN:
                    res_text = "PLAYER 1 WIN!"
                    res_color = COLOR_GREEN
                elif context.last_result == GameResult.PLAYER2_WIN:
                    res_text = "PLAYER 2 WIN!" if mode.is_pvp else "PC WIN!"
                    res_color = COLOR_RED
                elif context.last_result == GameResult.DRAW:
                    res_text = "DRAW"
                    res_color = COLOR_BLUE
                else:
                    res_text = "NO HAND DETECTED"
                    res_color = COLOR_GRAY

                banner_top = center_y - 75
                banner_bottom = center_y + 65
                self._draw_overlay_rect(
                    frame, 0, banner_top, w, banner_bottom, COLOR_BG_DARK, alpha=0.85
                )
                cv2.line(
                    frame, (0, banner_top), (w, banner_top), res_color, 2, cv2.LINE_AA
                )
                cv2.line(
                    frame,
                    (0, banner_bottom),
                    (w, banner_bottom),
                    res_color,
                    2,
                    cv2.LINE_AA,
                )

                self._draw_centered_text(
                    frame,
                    res_text,
                    center_x,
                    center_y - 25,
                    font_face=self.FONT_HEAD,
                    scale=1.05,
                    color=res_color,
                    thickness=2,
                )

                self._overlay_icon(
                    frame,
                    context.p1_final_gesture,
                    center_x=center_x - 75,
                    center_y=center_y + 22,
                    size=58,
                )
                self._draw_centered_text(
                    frame,
                    "VS",
                    center_x=center_x,
                    y=center_y + 28,
                    font_face=self.FONT_HEAD,
                    scale=0.6,
                    color=COLOR_WHITE,
                    thickness=2,
                )
                self._overlay_icon(
                    frame,
                    context.p2_final_gesture,
                    center_x=center_x + 75,
                    center_y=center_y + 22,
                    size=58,
                )

            case GameState.MATCH_OVER:
                winner = (
                    "PLAYER 1"
                    if context.p1_score > context.p2_score
                    else ("PLAYER 2" if mode.is_pvp else "PC")
                )
                self._draw_overlay_rect(
                    frame, 0, center_y - 65, w, center_y + 65, COLOR_BG_DARK, alpha=0.88
                )
                cv2.line(frame, (0, center_y - 65), (w, center_y - 65), COLOR_AMBER, 2)
                cv2.line(frame, (0, center_y + 65), (w, center_y + 65), COLOR_AMBER, 2)

                self._draw_centered_text(
                    frame,
                    f"MATCH WINNER: {winner}!",
                    center_x,
                    center_y - 8,
                    font_face=self.FONT_HEAD,
                    scale=1.1,
                    color=COLOR_AMBER,
                    thickness=2,
                )
                self._draw_centered_text(
                    frame,
                    "Press [R] to Start a New Match",
                    center_x,
                    center_y + 35,
                    scale=0.52,
                    color=COLOR_WHITE,
                    thickness=1,
                )

    def render(
        self,
        frame: np.ndarray,
        context: GameContext,
        detections: list[Detection],
        mode: GameMode,
        fps: float,
    ) -> None:
        self.draw_detections(frame, detections)
        self.draw_hud(frame, context, mode, fps)
        self.draw_state_overlays(frame, context, mode)
