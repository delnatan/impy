from datetime import datetime

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)


class MetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.resize(760, 560)

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select an item to see full details")
        self.details.setMinimumHeight(120)

        splitter.addWidget(self.tree)
        splitter.addWidget(self.details)
        splitter.setSizes([420, 140])

        layout.addWidget(splitter)

        self.populate_tree(metadata or {})

    def populate_tree(self, metadata):
        self.tree.clear()

        summary = self._build_summary_section(metadata)
        if summary:
            self._add_mapping_section("Summary", summary)

        timing = self._build_timing_section(metadata)
        if timing:
            self._add_mapping_section("Timing", timing)

        self._add_mapping_section("Raw Metadata", metadata)

        self.tree.expandToDepth(1)
        self.tree.resizeColumnToContents(0)

    def _add_mapping_section(self, title, mapping):
        root = QTreeWidgetItem(self.tree, [title, ""])
        root.setExpanded(True)
        self._add_value(root, mapping)

    def _add_value(self, parent, value):
        if isinstance(value, dict):
            for k, v in value.items():
                child = QTreeWidgetItem(parent, [str(k), ""])
                self._add_value(child, v)
            return

        if isinstance(value, np.ndarray):
            desc = f"ndarray shape={value.shape}, dtype={value.dtype}"
            if np.issubdtype(value.dtype, np.number) and value.size > 0:
                with np.errstate(invalid="ignore"):
                    vmin = np.nanmin(value)
                    vmax = np.nanmax(value)
                desc += f", min={vmin}, max={vmax}"
            parent.setText(1, desc)
            parent.setData(1, Qt.UserRole, desc)
            return

        if isinstance(value, (list, tuple)):
            n = len(value)
            parent.setText(1, f"{type(value).__name__} ({n})")
            parent.setData(1, Qt.UserRole, str(value))

            preview_limit = 30
            for idx, item in enumerate(value[:preview_limit]):
                child = QTreeWidgetItem(parent, [f"[{idx}]", ""])
                self._add_value(child, item)
            if n > preview_limit:
                QTreeWidgetItem(parent, ["...", f"{n - preview_limit} more items"])
            return

        text = self._format_scalar(value)
        visible = text if len(text) <= 140 else f"{text[:137]}..."
        parent.setText(1, visible)
        parent.setData(1, Qt.UserRole, text)
        if len(text) > len(visible):
            parent.setToolTip(1, text)

    def _on_current_item_changed(self, current, _previous):
        if current is None:
            self.details.clear()
            return

        key = current.text(0)
        raw = current.data(1, Qt.UserRole)
        value = raw if raw is not None else current.text(1)
        self.details.setPlainText(f"{key}\n\n{value}")

    def _build_summary_section(self, metadata):
        summary = {}
        for key in ("filename", "shape", "dtype", "scale", "is_rgb", "raw_shape"):
            if key in metadata:
                summary[key] = metadata[key]
        return summary

    def _build_timing_section(self, metadata):
        timing = {}

        timestamps = self._extract_timestamps(metadata)
        if timestamps:
            valid = [t for t in timestamps if t is not None]
            timing["timestamps"] = f"{len(valid)} / {len(timestamps)} valid"

            if valid:
                timing["acquisition_start"] = min(valid)
                timing["acquisition_end"] = max(valid)

            if len(valid) >= 2:
                sorted_ts = sorted(valid)
                dt_sec = [
                    (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
                    for i in range(len(sorted_ts) - 1)
                ]
                if dt_sec:
                    timing["frame_interval_mean_s"] = float(np.mean(dt_sec))
                    timing["frame_interval_min_s"] = float(np.min(dt_sec))
                    timing["frame_interval_max_s"] = float(np.max(dt_sec))
                    timing["frame_intervals_s"] = dt_sec

        exposure_map = self._extract_exposure_info(metadata)
        if exposure_map:
            timing["exposure_times_s"] = exposure_map

        return timing

    def _extract_timestamps(self, metadata):
        candidates = metadata.get("timestamps")
        if not isinstance(candidates, (list, tuple)):
            return []

        parsed = []
        for item in candidates:
            if item is None:
                parsed.append(None)
                continue
            if isinstance(item, datetime):
                parsed.append(item)
                continue
            if isinstance(item, np.datetime64):
                try:
                    ts = np.datetime_as_string(item, unit="us")
                    parsed.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                    continue
                except Exception:
                    parsed.append(None)
                    continue
            if isinstance(item, str):
                txt = item.strip()
                fmts = (
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                )
                dt = None
                for fmt in fmts:
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

    def _extract_exposure_info(self, metadata):
        channels = metadata.get("channels")
        if not isinstance(channels, (list, tuple)):
            return {}

        exposure = {}
        for i, ch in enumerate(channels):
            if not isinstance(ch, dict):
                continue

            value = ch.get("exposure_time")
            if value in (None, ""):
                continue

            name = ch.get("name") or f"Channel {i}"
            exposure[name] = value

        return exposure

    def _format_scalar(self, value):
        if isinstance(value, datetime):
            return value.isoformat(sep=" ")
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)
