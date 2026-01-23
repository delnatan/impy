import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class MetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        self.populate_table(metadata)

    def populate_table(self, metadata):
        self.table.setRowCount(0)
        for key, value in metadata.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Key
            k_item = QTableWidgetItem(str(key))
            k_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 0, k_item)

            # Value
            v_str = str(value)
            # If value is a long list/array, truncate it?
            if (
                isinstance(value, (list, tuple, np.ndarray))
                and len(value) > 10
            ):
                v_str = f"{type(value).__name__} shape={np.shape(value)}"

            v_item = QTableWidgetItem(v_str)
            v_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 1, v_item)
