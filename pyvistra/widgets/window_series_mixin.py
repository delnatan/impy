"""WindowSeriesMixin — shared "compare a shape against several windows"
scaffolding for :class:`~pyvistra.widgets.line_profile.LineProfileDialog`
and :class:`~pyvistra.widgets.radial_profile_dialog.RadialProfileDialog`.

Both dialogs let the user pick a shape on one "source" window, then build a
list of other windows ("series") to sample the same shape against, with
per-series visibility/channel/color and live updates as windows
open/close/activate. That bookkeeping — not the shape-specific sampling or
plotting — is what's shared here. The two dialogs have already drifted in a
few small, deliberate ways (line profile draws overlays and refreshes on
window activation; radial profile does neither), so this mixin uses
template-method hooks at exactly those points rather than forcing identical
behavior. Each hook's default matches whichever dialog's current behavior is
more permissive/simpler; the other dialog overrides it. See each hook's
docstring for which dialog currently relies on the override.

Host contract — the class mixing this in must, before calling
``_init_window_series()``:
  - set ``self.series_config`` (``OrderedDict``), ``self._computed_series``
    (``list``), ``self.active_window``, ``self.source_window``,
    ``self._is_shutting_down`` (``bool``)
  - build (via its own ``_setup_ui()``) ``self.series_list`` (QListWidget),
    ``self.all_channels_cb`` (QCheckBox), ``self.series_channel_spin``
    (QSpinBox), ``self.status_label`` (QLabel), ``self.profile_widget``
    (anything with ``.clear()``)
  - implement ``self._refresh_profiles()`` and
    ``self._unsubscribe_from_shape_source()``
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QInputDialog, QListWidgetItem

from ..ui.manager import manager
from .series_colors import FALLBACK_COLORS, _to_qcolor


class WindowSeriesMixin:
    def _init_window_series(self):
        """Call once from ``__init__``, after ``_setup_ui()``."""
        self._connected_windows = set()
        manager.window_registered.connect(self._on_window_registered)
        for window in manager.get_all().values():
            self._connect_window(window)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------
    def _on_window_registered(self, window):
        if self._is_shutting_down:
            return
        self._connect_window(window)

    def _connect_window(self, window):
        if window in self._connected_windows:
            return
        window.window_activated.connect(self._on_window_activated)
        window.window_closing.connect(self._on_window_closing)
        self._connect_extra_series_signals(window)
        self._connected_windows.add(window)

    def _disconnect_window(self, window):
        if window not in self._connected_windows:
            return
        try:
            window.window_activated.disconnect(self._on_window_activated)
            window.window_closing.disconnect(self._on_window_closing)
            self._disconnect_extra_series_signals(window)
        except (TypeError, RuntimeError):
            pass
        self._connected_windows.discard(window)

    def _connect_extra_series_signals(self, window):
        """Hook: connect signals beyond window_activated/window_closing.

        Default: none. LineProfileDialog overrides to also connect
        roi_selection_changed and (conditionally) view_changed;
        RadialProfileDialog overrides to connect view_changed
        unconditionally.
        """

    def _disconnect_extra_series_signals(self, window):
        """Hook: mirror of :meth:`_connect_extra_series_signals`."""

    def _on_window_activated(self, window):
        if self._is_shutting_down:
            return
        self.active_window = window
        self._on_active_window_changed()

    def _on_active_window_changed(self):
        """Hook: called after ``self.active_window`` updates.

        Default: no-op (matches RadialProfileDialog). LineProfileDialog
        overrides to refresh immediately when it already has shape data.
        """

    def _on_window_closing(self, window):
        self._disconnect_window(window)

        wid = getattr(window, "window_id", None)
        if wid in self.series_config:
            self.series_config.pop(wid, None)
            self._refresh_series_list()
            self._refresh_profiles()

        if window == self.active_window:
            self.active_window = None
        if window == self.source_window:
            self.source_window = None
            self._on_source_window_closed()
            self._unsubscribe_from_shape_source()

    def _on_source_window_closed(self):
        """Hook: extra cleanup when the shape-source window closes.

        Default: no-op (matches LineProfileDialog, which leaves
        ``current_line_data`` set so a stale profile isn't silently
        discarded). RadialProfileDialog overrides to clear
        ``current_circle_data``.
        """

    def _on_view_changed(self, window):
        if self._is_shutting_down or not self._has_shape_source():
            return
        wid = getattr(window, "window_id", None)
        source_wid = getattr(self.source_window, "window_id", None)
        if wid in self.series_config or wid == source_wid:
            self._refresh_profiles()

    def _has_shape_source(self) -> bool:
        """Hook: whether a shape source is currently set (i.e.
        ``current_line_data``/``current_circle_data`` is not None)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Series management
    # ------------------------------------------------------------------
    def _add_active_window_series(self):
        if self.active_window is None:
            self.status_label.setText("No active window")
            return
        self._add_series_for_window(self.active_window)
        self._refresh_profiles()

    def _add_window_series_dialog(self):
        windows = self._eligible_series_windows()
        labels = [f"[{w.window_id}] {w.windowTitle()}" for w in windows]
        if not windows:
            self.status_label.setText("No image windows available")
            return

        item, ok = QInputDialog.getItem(
            self, "Add Window", "Select window to compare:", labels, 0, False,
        )
        if not ok or not item:
            return

        idx = labels.index(item)
        self._add_series_for_window(windows[idx])
        self._refresh_profiles()

    def _eligible_series_windows(self):
        """Hook: which open windows are offered in "Add Window...".

        Default: every window the manager knows about (matches
        RadialProfileDialog). LineProfileDialog overrides to only offer
        windows with an ``roi_added`` signal.
        """
        return list(manager.get_all().values())

    def _add_series_for_window(self, window, channel_idx=None):
        wid = window.window_id
        if wid in self.series_config:
            return

        if channel_idx is None:
            channel_idx = int(getattr(window, "c_idx", 0))
        num_channels = int(getattr(window, "C", 1))
        channel_idx = int(np.clip(channel_idx, 0, max(0, num_channels - 1)))

        color = self._get_window_channel_color(window, channel_idx, len(self.series_config))

        self.series_config[wid] = {
            "window": window,
            "channel": channel_idx,
            "visible": True,
            "label": f"[{window.window_id}] {window.windowTitle()}",
            "color": color,
        }
        self._refresh_series_list(select_wid=wid)

    def _remove_selected_series(self):
        item = self.series_list.currentItem()
        if item is None:
            return

        wid = item.data(Qt.UserRole)
        cfg = self.series_config.pop(wid, None)
        if cfg is not None:
            self._on_series_removed(cfg)

        self._refresh_series_list()
        self._refresh_profiles()

    def _on_series_removed(self, cfg):
        """Hook: called after a series is removed via "Remove Selected".

        Default: no-op (matches RadialProfileDialog). LineProfileDialog
        overrides to hide that window's profile-path overlay.
        """

    def _clear_series(self):
        self._on_series_cleared_pre()
        self.series_config.clear()
        self._computed_series = []
        self._reset_shape_source_data()
        self.source_window = None
        self._unsubscribe_from_shape_source()

        self._refresh_series_list()
        self.profile_widget.clear()
        self.status_label.setText(self._empty_status_text())

    def _on_series_cleared_pre(self):
        """Hook: runs before series_config is cleared.

        Default: no-op (matches RadialProfileDialog). LineProfileDialog
        overrides to hide all profile-path overlays.
        """

    def _reset_shape_source_data(self):
        """Hook: reset current_line_data/current_circle_data to None."""
        raise NotImplementedError

    def _empty_status_text(self) -> str:
        """Hook: status text shown once there's no shape source."""
        return "Select a shape to start"

    def _refresh_series_list(self, select_wid=None):
        selected_wid = None
        current_item = self.series_list.currentItem()
        if current_item is not None:
            selected_wid = current_item.data(Qt.UserRole)

        self.series_list.blockSignals(True)
        self.series_list.clear()

        all_ch = self.all_channels_cb.isChecked()
        for wid, cfg in self.series_config.items():
            label = cfg.get("label", f"[{wid}] Window")
            if all_ch:
                n_ch = self._num_channels(cfg.get("window")) if cfg.get("window") else 1
                ch_text = f"All ({n_ch}ch)"
            else:
                ch_text = f"Ch{int(cfg.get('channel', 0)) + 1}"
            item = QListWidgetItem(f"{label} | {ch_text}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if cfg.get("visible", True) else Qt.Unchecked)
            item.setData(Qt.UserRole, wid)
            self.series_list.addItem(item)

        target_wid = select_wid if select_wid is not None else selected_wid
        if target_wid is not None:
            for i in range(self.series_list.count()):
                item = self.series_list.item(i)
                if item.data(Qt.UserRole) == target_wid:
                    self.series_list.setCurrentItem(item)
                    break

        self.series_list.blockSignals(False)
        self._on_series_selection_changed()

    def _on_series_item_changed(self, item):
        wid = item.data(Qt.UserRole)
        if wid not in self.series_config:
            return

        self.series_config[wid]["visible"] = item.checkState() == Qt.Checked
        self._refresh_profiles()

    def _on_series_selection_changed(self):
        item = self.series_list.currentItem()
        if item is None:
            self.series_channel_spin.blockSignals(True)
            self.series_channel_spin.setRange(1, 1)
            self.series_channel_spin.setValue(1)
            self.series_channel_spin.blockSignals(False)
            self.series_channel_spin.setEnabled(False)
            return

        wid = item.data(Qt.UserRole)
        if wid not in self.series_config:
            return

        cfg = self.series_config[wid]
        window = cfg["window"]
        n_channels = int(getattr(window, "C", 1))
        ch = int(cfg.get("channel", 0)) + 1

        self.series_channel_spin.blockSignals(True)
        self.series_channel_spin.setRange(1, max(1, n_channels))
        self.series_channel_spin.setValue(int(np.clip(ch, 1, max(1, n_channels))))
        self.series_channel_spin.blockSignals(False)
        self.series_channel_spin.setEnabled(
            n_channels > 1 and not self.all_channels_cb.isChecked()
        )

    def _on_all_channels_toggled(self, checked):
        self.series_channel_spin.setEnabled(not checked and self.series_list.currentItem() is not None)
        self._refresh_series_list()
        self._refresh_profiles()

    def _on_selected_channel_changed(self, value):
        item = self.series_list.currentItem()
        if item is None:
            return

        wid = item.data(Qt.UserRole)
        if wid not in self.series_config:
            return

        cfg = self.series_config[wid]
        window = cfg["window"]
        n_channels = int(getattr(window, "C", 1))
        new_channel = int(np.clip(value - 1, 0, max(0, n_channels - 1)))

        cfg["channel"] = new_channel
        cfg["color"] = self._get_window_channel_color(window, new_channel, 0)

        self._refresh_series_list(select_wid=wid)
        self._refresh_profiles()

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _num_channels(window):
        cache = window.renderer.current_slice_cache
        if cache is not None and cache.ndim == 3:
            return cache.shape[0]
        return int(getattr(window, "C", 1))

    def _get_window_channel_color(self, window, channel_idx, fallback_idx):
        colors = getattr(window.renderer, "channel_colors", [])
        if channel_idx < len(colors):
            return _to_qcolor(colors[channel_idx], QColor(FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])).name()
        return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._is_shutting_down:
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()

    def cleanup(self):
        self._is_shutting_down = True
        self._on_cleanup_extra()

        try:
            manager.window_registered.disconnect(self._on_window_registered)
        except (TypeError, RuntimeError):
            pass

        for window in list(self._connected_windows):
            self._disconnect_window(window)

        self._unsubscribe_from_shape_source()
        self.active_window = None
        self.source_window = None
        self._reset_shape_source_data()

    def _on_cleanup_extra(self):
        """Hook: extra cleanup at the very start of ``cleanup()``.

        Default: no-op (matches RadialProfileDialog). LineProfileDialog
        overrides to hide all profile-path overlays.
        """
