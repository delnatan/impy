from datetime import datetime

import numpy as np
from qtpy.QtCore import QObject, QTimer


class PlaybackController(QObject):
    """
    Drives timelapse playback: timer scheduling, fps state, and syncing the
    Play button / fps spinbox / realtime button.

    Decoupled from any particular widget via three callables the caller
    supplies at construction: how many frames there are, what the current
    frame is, and how to move to a new one. This keeps the controller usable
    wherever a "t index" exists, not just on ImageWindow's slider layout.
    """

    def __init__(self, *, frame_count, time_index, advance_time, timestamps, parent=None):
        super().__init__(parent)
        self._frame_count = frame_count
        self._time_index = time_index
        self._advance_time = advance_time
        self._timestamps = timestamps

        self.fps = 5.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance_frame)

        self.play_button = None
        self.fps_spin = None
        self.realtime_button = None

        realtime_fps = self.estimate_realtime_fps()
        if realtime_fps is not None:
            self.fps = realtime_fps

    def bind_widgets(self, play_button, fps_spin, realtime_button):
        """Wire the Play/fps/Realtime widgets and sync their initial state."""
        self.play_button = play_button
        self.fps_spin = fps_spin
        self.realtime_button = realtime_button
        self.refresh_realtime_ui()

    @property
    def is_playing(self):
        return self.play_button is not None and self.play_button.isChecked()

    def toggle(self, checked):
        if checked:
            self.start()
        else:
            self.stop()

    def start(self):
        if self._frame_count() <= 1:
            if self.play_button is not None:
                self.play_button.blockSignals(True)
                self.play_button.setChecked(False)
                self.play_button.blockSignals(False)
            return
        if self.play_button is not None:
            self.play_button.setText("Pause")
        self._schedule_next_step()

    def stop(self):
        self._timer.stop()
        if self.play_button is not None:
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.setText("Play")
            self.play_button.blockSignals(False)

    def on_time_changed(self):
        """
        Call whenever the time index moves, regardless of the reason (our own
        timer, slider drag, scripting) -- reschedules the next step so fps
        stays correct if the index was nudged mid-playback.
        """
        if self.is_playing:
            self._schedule_next_step()

    def set_fps(self, fps):
        self.fps = max(0.01, float(fps))
        if self.is_playing:
            self._schedule_next_step()

    def set_realtime_fps(self):
        realtime_fps = self.estimate_realtime_fps()
        if realtime_fps is None:
            return
        self.fps = realtime_fps
        if self.fps_spin is not None:
            self.fps_spin.setValue(realtime_fps)

    def refresh_realtime_ui(self):
        realtime_fps = self.estimate_realtime_fps()
        if self.realtime_button is not None:
            self.realtime_button.setEnabled(realtime_fps is not None)
            if realtime_fps is None:
                self.realtime_button.setToolTip(
                    "Realtime FPS unavailable (missing/invalid timestamps)"
                )
            else:
                self.realtime_button.setToolTip(
                    f"Set to realtime ({realtime_fps:.3g} fps)"
                )

    def estimate_realtime_fps(self):
        timestamps = self._parsed_timestamps()
        if len(timestamps) < 2:
            return None

        deltas = []
        for t0, t1 in zip(timestamps, timestamps[1:]):
            if t0 is None or t1 is None:
                continue
            try:
                dt = float((t1 - t0).total_seconds())
            except Exception:
                continue
            if dt > 0:
                deltas.append(dt)

        if not deltas:
            return None

        mean_dt = float(np.mean(deltas))
        if mean_dt <= 0:
            return None

        return max(0.01, min(120.0, 1.0 / mean_dt))

    def _advance_frame(self):
        if not self.is_playing or self._frame_count() <= 1:
            return

        next_idx = (self._time_index() + 1) % self._frame_count()
        self._advance_time(next_idx)

        self._schedule_next_step()

    def _schedule_next_step(self):
        self._timer.stop()
        if not self.is_playing or self._frame_count() <= 1:
            return

        delay_ms = int(1000.0 / max(0.01, self.fps))
        self._timer.start(max(1, delay_ms))

    def _parsed_timestamps(self):
        timestamps = self._timestamps()
        if not isinstance(timestamps, (list, tuple)):
            return []

        parsed = []
        for item in timestamps:
            if item is None:
                parsed.append(None)
                continue
            if isinstance(item, datetime):
                parsed.append(item)
                continue
            if isinstance(item, np.datetime64):
                try:
                    iso = np.datetime_as_string(item, unit="us")
                    parsed.append(
                        datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    )
                except Exception:
                    parsed.append(None)
                continue
            if isinstance(item, str):
                txt = item.strip()
                dt = None
                for fmt in (
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                ):
                    try:
                        dt = datetime.strptime(txt, fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    try:
                        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
                    except ValueError:
                        dt = None
                parsed.append(dt)
                continue

            parsed.append(None)

        return parsed
