"""
Axes reordering dialog for ambiguous TIFF dimensions.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class AxesDialog(QDialog):
    """
    Dialog for reordering image axes when dimension interpretation is ambiguous.

    Shows the original shape (before 5D normalization) and lets the user assign
    T, Z, or C meaning to each non-spatial dimension.
    """

    def __init__(self, raw_shape, parent=None):
        """
        Args:
            raw_shape: Original array shape before 5D normalization (e.g., (10, 3, 512, 512))
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Reorder Axes")
        self.raw_shape = raw_shape
        self.combos = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Shape info
        shape_str = " x ".join(str(s) for s in self.raw_shape)
        layout.addWidget(QLabel(f"Current shape: ({shape_str})"))
        layout.addSpacing(10)

        # Number of non-spatial dimensions
        ndim = len(self.raw_shape)
        n_extra = ndim - 2  # Dimensions beyond Y, X

        if n_extra <= 0:
            # 2D image, nothing to configure
            layout.addWidget(QLabel("2D image - no axes to reorder."))
        else:
            # Create combo boxes for each non-spatial dimension
            form = QFormLayout()

            for i in range(n_extra):
                size = self.raw_shape[i]
                combo = QComboBox()
                combo.addItems(["T (Time)", "Z (Depth)", "C (Channel)"])

                # Default assignment based on heuristics
                if n_extra == 1:
                    # 3D: default to Z
                    combo.setCurrentIndex(1)  # Z
                elif n_extra == 2:
                    # 4D: default to Z, C
                    combo.setCurrentIndex(1 if i == 0 else 2)
                elif n_extra == 3:
                    # 5D: default to T, Z, C
                    combo.setCurrentIndex(i)

                self.combos.append(combo)
                form.addRow(f"Dimension {i} (size {size}):", combo)

            layout.addLayout(form)
            layout.addWidget(QLabel("(Y, X are fixed as last two dimensions)"))

        layout.addSpacing(10)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok
        )
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_dims_string(self):
        """
        Returns the dimension string based on user selection.

        Returns:
            str: Dimension string like 'tyx', 'zyx', 'zcyx', 'tzyx', 'tzcyx'
        """
        if not self.combos:
            return "yx"

        dim_map = {0: "t", 1: "z", 2: "c"}
        dims = ""

        for combo in self.combos:
            idx = combo.currentIndex()
            dims += dim_map[idx]

        dims += "yx"
        return dims
