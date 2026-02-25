"""
Gel Image Analyzer Widget

Provides molecular weight/size estimation from gel electrophoresis images.
Works with both protein gels (kDa) and DNA gels (bp).
Uses LaneROI to define lanes with integrated peak markers.
"""

import json
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks

from ..rois import LaneROI, RectangleROI


def get_builtin_ladders_path():
    """Return path to the built-in ladders.json file."""
    return Path(__file__).resolve().parents[1] / "_resources" / "ladders.json"


def load_ladders(path=None):
    """Load ladder presets from JSON file."""
    if path is None:
        path = get_builtin_ladders_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ladders", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ladders: {e}")
        return []


class GelAnalyzerWidget(QWidget):
    """
    Widget for analyzing gel electrophoresis images.

    Detects bands in lanes (LaneROI), uses one lane as a
    molecular weight ladder, and estimates MW of bands in other lanes.
    Supports both protein (kDa) and DNA (bp) gels.
    """

    def __init__(self, annotation_manager):
        super().__init__()
        self.annotation_manager = annotation_manager
        self.ladders = load_ladders()
        self._lane_rois = []  # Track LaneROIs we've created/converted
        self._std_idx = None  # Index of standard lane ROI

        # Calibration state
        self._calibration = None  # {std_positions, log_sizes, unit}

        self.setWindowTitle("Gel Analyzer")
        self.resize(320, 560)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Ladder Preset Section
        ladder_group = QGroupBox("Ladder Preset")
        ladder_layout = QVBoxLayout(ladder_group)

        self.ladder_combo = QComboBox()
        self.ladder_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        for ladder in self.ladders:
            unit = ladder.get("unit", "kDa")
            display_name = f"{ladder['name']} ({unit})"
            self.ladder_combo.addItem(display_name, userData=ladder)
        ladder_layout.addWidget(self.ladder_combo)

        btn_load_custom = QPushButton("Load Custom Ladder...")
        btn_load_custom.clicked.connect(self._load_custom_ladder)
        ladder_layout.addWidget(btn_load_custom)

        layout.addWidget(ladder_group)

        # Lane Selection Section
        lane_group = QGroupBox("Standard Lane")
        lane_layout = QHBoxLayout(lane_group)

        lane_layout.addWidget(QLabel("Ladder ROI:"))
        self.lane_combo = QComboBox()
        self.lane_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lane_layout.addWidget(self.lane_combo)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_lanes)
        lane_layout.addWidget(btn_refresh)

        layout.addWidget(lane_group)

        # Peak Detection Section
        detect_group = QGroupBox("Peak Detection")
        detect_layout = QVBoxLayout(detect_group)

        # Invert intensity checkbox
        self.invert_check = QCheckBox("Invert intensity (dark bands)")
        self.invert_check.setChecked(True)
        self.invert_check.setToolTip(
            "Check for Coomassie/silver stains (dark bands on light background)\n"
            "Uncheck for fluorescent stains (bright bands on dark background)"
        )
        detect_layout.addWidget(self.invert_check)

        prom_layout = QHBoxLayout()
        prom_layout.addWidget(QLabel("Prominence:"))
        self.prominence_spin = QSpinBox()
        self.prominence_spin.setRange(100, 50000)
        self.prominence_spin.setValue(1000)
        self.prominence_spin.setSingleStep(100)
        prom_layout.addWidget(self.prominence_spin)
        detect_layout.addLayout(prom_layout)

        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Min Distance (px):"))
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(5)
        dist_layout.addWidget(self.distance_spin)
        detect_layout.addLayout(dist_layout)

        btn_detect = QPushButton("Detect Peaks")
        btn_detect.clicked.connect(self._detect_peaks)
        detect_layout.addWidget(btn_detect)

        layout.addWidget(detect_group)

        # Actions Section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        # Calibrate and Apply buttons in a row
        calib_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.clicked.connect(self._calibrate)
        self.btn_calibrate.setToolTip(
            "Create calibration curve from ladder lane markers"
        )
        calib_row.addWidget(self.btn_calibrate)

        self.btn_apply = QPushButton("Apply Calibration")
        self.btn_apply.clicked.connect(self._apply_calibration)
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip("Estimate MW for sample lanes using calibration")
        calib_row.addWidget(self.btn_apply)
        actions_layout.addLayout(calib_row)

        btn_clear = QPushButton("Clear All Markers")
        btn_clear.clicked.connect(self._clear_markers)
        actions_layout.addWidget(btn_clear)

        layout.addWidget(actions_group)

        # Display Options
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.show_labels_check = QCheckBox("Show peak labels")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self._toggle_labels)
        display_layout.addWidget(self.show_labels_check)

        self.show_borders_check = QCheckBox("Show lane borders")
        self.show_borders_check.setChecked(True)
        self.show_borders_check.stateChanged.connect(self._toggle_borders)
        display_layout.addWidget(self.show_borders_check)

        tip_label = QLabel(
            "Tip: Shift+Click to add band, Ctrl/Cmd+Click to remove\n"
            "     Drag lines to adjust positions"
        )
        tip_label.setStyleSheet("color: gray; font-size: 10px;")
        display_layout.addWidget(tip_label)

        layout.addWidget(display_group)

        # Results Section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(100)
        results_layout.addWidget(self.results_text)

        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(self._copy_results)
        results_layout.addWidget(btn_copy)

        btn_export = QPushButton("Export CSV...")
        btn_export.clicked.connect(self._export_csv)
        results_layout.addWidget(btn_export)

        layout.addWidget(results_group)

    def showEvent(self, event):
        """Refresh lane list when shown."""
        super().showEvent(event)
        self._refresh_lanes()

    def closeEvent(self, event):
        """Clean up when closed."""
        self._clear_markers()
        super().closeEvent(event)

    def _toggle_labels(self, state):
        """Toggle peak label visibility."""
        visible = state == Qt.Checked.value
        window = self.annotation_manager.active_window
        if window:
            for roi in window.rois:
                if isinstance(roi, LaneROI):
                    roi.set_marker_labels_visible(visible)
        self._update_canvas()

    def _toggle_borders(self, state):
        """Toggle lane border visibility."""
        show_border = state == Qt.Checked.value
        window = self.annotation_manager.active_window
        if window:
            for roi in window.rois:
                if isinstance(roi, LaneROI):
                    roi.show_border = show_border
        self._update_canvas()

    def _update_canvas(self):
        """Update the canvas."""
        window = self.annotation_manager.active_window
        if window:
            window.canvas.update()

    def _load_custom_ladder(self):
        """Load a custom ladder preset from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ladder Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Support both single ladder and list format
            if "ladders" in data:
                new_ladders = data["ladders"]
            elif "name" in data and ("sizes" in data or "weights_kda" in data):
                new_ladders = [data]
            else:
                print("Invalid ladder format")
                return

            for ladder in new_ladders:
                # Normalize old format
                if "weights_kda" in ladder and "sizes" not in ladder:
                    ladder["sizes"] = ladder["weights_kda"]
                    ladder["unit"] = "kDa"
                self.ladders.append(ladder)
                unit = ladder.get("unit", "kDa")
                display_name = f"{ladder['name']} ({unit})"
                self.ladder_combo.addItem(display_name, userData=ladder)

            self.ladder_combo.setCurrentIndex(self.ladder_combo.count() - 1)
            print(f"Loaded {len(new_ladders)} ladder(s) from {path}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading ladder file: {e}")

    def _refresh_lanes(self):
        """Refresh the lane combo box with current Rectangle/Lane ROIs."""
        self.lane_combo.clear()

        window = self.annotation_manager.active_window
        if not window:
            return

        for i, roi in enumerate(window.rois):
            if isinstance(roi, (RectangleROI, LaneROI)):
                self.lane_combo.addItem(f"{i}: {roi.name}", userData=i)

    def _get_current_image(self):
        """Get the current 2D grayscale image from the active window."""
        window = self.annotation_manager.active_window
        if not window:
            return None

        cache = window.renderer.current_slice_cache
        if cache is None:
            return None

        # Convert to grayscale if multi-channel
        if cache.ndim == 3:
            gray = cache.mean(axis=0)
        else:
            gray = cache

        return gray

    def _convert_to_lane_roi(self, window, roi_idx, roi):
        """Convert a RectangleROI to LaneROI."""
        if isinstance(roi, LaneROI):
            return roi  # Already a LaneROI

        # Create new LaneROI with same bounds
        lane = LaneROI(window.view, name=roi.name)
        lane.update(roi.data["p1"], roi.data["p2"])

        # Replace in window's ROI list
        window.rois[roi_idx] = lane

        # Remove old ROI visuals
        roi.remove()

        # Emit signal for annotation manager
        window.roi_removed.emit(roi)
        window.roi_added.emit(lane)

        return lane

    def _detect_peaks(self):
        """Detect peaks in all lanes and update markers (no MW calculation)."""
        # Clear previous state
        self._clear_markers()
        self._calibration = None
        self.btn_apply.setEnabled(False)

        window = self.annotation_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        # Get standard lane index
        self._std_idx = self.lane_combo.currentData()
        if self._std_idx is None:
            self.results_text.setText("No standard lane selected")
            return

        # Get image
        gray = self._get_current_image()
        if gray is None:
            self.results_text.setText("No image data available")
            return

        # Invert intensity if needed
        if self.invert_check.isChecked():
            max_val = gray.max()
            gray_proc = max_val - gray
        else:
            gray_proc = gray

        # Get detection parameters
        prominence = self.prominence_spin.value()
        min_distance = self.distance_spin.value()

        # Find and convert Rectangle ROIs to Lane ROIs
        self._lane_rois = []
        roi_indices = []

        for i, roi in enumerate(window.rois):
            if isinstance(roi, (RectangleROI, LaneROI)):
                lane = self._convert_to_lane_roi(window, i, roi)
                self._lane_rois.append(lane)
                roi_indices.append(i)

        if not self._lane_rois:
            self.results_text.setText("No rectangle/lane ROIs found")
            return

        # Detect peaks in each lane
        total_peaks = 0
        for lane, roi_idx in zip(self._lane_rois, roi_indices):
            region = lane.get_region(gray_proc)
            if region.size == 0:
                continue

            # Get 1D profile (mean across lane width)
            profile = region.mean(axis=1)

            # Find peaks
            peaks, _ = find_peaks(
                profile, prominence=prominence, distance=min_distance
            )

            # Determine color based on whether this is the ladder
            is_ladder = roi_idx == self._std_idx
            color = LaneROI.LADDER_COLOR if is_ladder else LaneROI.SAMPLE_COLOR

            # Create marker data (no labels yet - that comes from calibration)
            marker_data = []
            for peak_y in peaks:
                marker_data.append(
                    {"y_local": float(peak_y), "label": "", "color": color}
                )

            # Set markers on lane
            lane.set_markers(marker_data)
            lane.locked = True
            lane.set_markers_changed_callback(self._on_markers_adjusted)

            # Apply current display settings
            lane.set_marker_labels_visible(self.show_labels_check.isChecked())
            lane.show_border = self.show_borders_check.isChecked()

            total_peaks += len(peaks)

        window.canvas.update()

        # Update annotation manager list
        self.annotation_manager.refresh_list()

        # Show detection summary
        results = [
            f"Detected {total_peaks} peaks across {len(self._lane_rois)} lanes.",
            "",
            "Next steps:",
            "  1. Adjust markers if needed (drag, Shift+Click, Ctrl+Click)",
            "  2. Click 'Calibrate' to create calibration from ladder lane",
            "  3. Click 'Apply Calibration' to estimate MW for sample lanes",
        ]
        self.results_text.setText("\n".join(results))

    def _calibrate(self):
        """Create calibration curve from ladder lane markers."""
        window = self.annotation_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        # Get ladder data
        ladder_data = self.ladder_combo.currentData()
        if ladder_data is None:
            self.results_text.setText("No ladder preset selected")
            return

        ladder_sizes = ladder_data.get(
            "sizes", ladder_data.get("weights_kda", [])
        )
        unit = ladder_data.get("unit", "kDa")

        # Get standard lane index
        if self._std_idx is None:
            self._std_idx = self.lane_combo.currentData()
        if self._std_idx is None:
            self.results_text.setText("No standard lane selected")
            return

        # Find standard lane and its markers
        std_lane = None
        for i, roi in enumerate(window.rois):
            if i == self._std_idx and isinstance(roi, LaneROI):
                std_lane = roi
                break

        if std_lane is None:
            self.results_text.setText("Standard lane not found")
            return

        std_peaks = std_lane.get_marker_positions()
        if len(std_peaks) == 0:
            self.results_text.setText("No markers in standard lane")
            return

        # Match peaks to ladder sizes
        n_peaks = min(len(std_peaks), len(ladder_sizes))
        if n_peaks < 2:
            self.results_text.setText(
                f"Need at least 2 markers in standard lane "
                f"(found {len(std_peaks)})"
            )
            return

        # Create calibration curve (in log scale)
        std_positions = np.array(std_peaks[:n_peaks])
        log_sizes = np.log(np.array(ladder_sizes[:n_peaks]))

        # Store calibration
        self._calibration = {
            "std_positions": std_positions,
            "log_sizes": log_sizes,
            "unit": unit,
            "ladder_sizes": ladder_sizes[:n_peaks],
        }

        # Update ladder lane marker labels with sizes
        labels = [f"{sz} {unit}" for sz in ladder_sizes[:n_peaks]]
        # Pad with empty strings if there are extra markers
        labels.extend([""] * (len(std_peaks) - n_peaks))
        std_lane.update_marker_labels(labels)

        # Enable apply button
        self.btn_apply.setEnabled(True)

        # Show calibration summary
        results = [
            "Calibration created!",
            f"  Standard Lane: ROI {self._std_idx}",
            f"  Using {n_peaks} bands: {', '.join(str(s) for s in ladder_sizes[:n_peaks])} {unit}",
            "",
            "Click 'Apply Calibration' to estimate MW for sample lanes.",
        ]
        self.results_text.setText("\n".join(results))
        window.canvas.update()

    def _apply_calibration(self):
        """Apply calibration to estimate MW for sample lanes."""
        if self._calibration is None:
            self.results_text.setText("No calibration available. Click 'Calibrate' first.")
            return

        window = self.annotation_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        std_positions = self._calibration["std_positions"]
        log_sizes = self._calibration["log_sizes"]
        unit = self._calibration["unit"]

        results = []
        results.append(f"Standard Lane (ROI {self._std_idx}):")
        results.append(f"  Calibration: {len(std_positions)} bands")
        results.append("")

        # Calculate sizes for sample lanes
        for i, roi in enumerate(window.rois):
            if not isinstance(roi, LaneROI):
                continue
            if i == self._std_idx:
                continue

            peaks = roi.get_marker_positions()
            if len(peaks) == 0:
                results.append(f"Lane {i} ({roi.name}): No markers")
                continue

            # Interpolate in log space
            log_mw = np.interp(peaks, std_positions, log_sizes)
            mw = np.exp(log_mw)

            # Update marker labels
            labels = [f"~{w:.1f} {unit}" for w in mw]
            roi.update_marker_labels(labels)

            results.append(f"Lane {i} ({roi.name}):")
            mw_strs = [f"{w:.1f}" for w in mw]
            results.append(f"  Size ({unit}): {', '.join(mw_strs)}")

        self.results_text.setText("\n".join(results))
        window.canvas.update()

    def _on_markers_adjusted(self):
        """Called when markers are manually adjusted."""
        # If calibrated, re-apply calibration for live updates
        if self._calibration is not None:
            self._apply_calibration()

    def _clear_markers(self):
        """Clear all markers, unlock lanes, and reset calibration."""
        window = self.annotation_manager.active_window
        if not window:
            return

        for roi in window.rois:
            if isinstance(roi, LaneROI):
                roi.clear_markers()
                roi.locked = False
                roi.show_border = True

        self._lane_rois = []
        self._std_idx = None
        self._calibration = None
        self.btn_apply.setEnabled(False)
        window.canvas.update()

    def _copy_results(self):
        """Copy results to clipboard."""
        from qtpy.QtWidgets import QApplication

        text = self.results_text.toPlainText()
        QApplication.clipboard().setText(text)

    def _export_csv(self):
        """Export results as CSV file."""
        text = self.results_text.toPlainText()
        if not text:
            return

        # Get unit from current ladder
        ladder_data = self.ladder_combo.currentData()
        unit = ladder_data.get("unit", "kDa") if ladder_data else "kDa"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "gel_analysis.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        # Parse results and write CSV
        lines = text.strip().split("\n")
        with open(path, "w") as f:
            f.write(f"Lane,ROI_Name,Size_{unit}\n")
            current_lane = ""
            current_name = ""
            for line in lines:
                line = line.strip()
                if line.startswith("Lane"):
                    parts = line.rstrip(":").split(" ", 2)
                    if len(parts) >= 2:
                        current_lane = parts[1]
                        if len(parts) > 2 and "(" in parts[2]:
                            current_name = parts[2].strip("():")
                elif line.startswith("Size"):
                    mw_part = line.split(": ", 1)
                    if len(mw_part) == 2:
                        mw_values = mw_part[1].split(", ")
                        for mw in mw_values:
                            f.write(f"{current_lane},{current_name},{mw}\n")

        print(f"Results exported to {path}")


# Singleton instance
_gel_analyzer_instance = None


def get_gel_analyzer(annotation_manager):
    """Get or create the GelAnalyzer singleton."""
    global _gel_analyzer_instance
    if _gel_analyzer_instance is None:
        _gel_analyzer_instance = GelAnalyzerWidget(annotation_manager)
    return _gel_analyzer_instance


def show_gel_analyzer(annotation_manager):
    """Show the Gel Analyzer widget."""
    analyzer = get_gel_analyzer(annotation_manager)
    analyzer.show()
    analyzer.raise_()
    analyzer.activateWindow()
