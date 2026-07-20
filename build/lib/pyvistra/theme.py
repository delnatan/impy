DARK_THEME = """
/* Main Window & Background */
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-size: 12px;
}

/* Tooltips */
QToolTip {
    background-color: #333333;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 4px;
}

/* Buttons */
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #2563eb;
}

QPushButton:pressed {
    background-color: #1d4ed8;
}

QPushButton:disabled {
    background-color: #4b5563;
    color: #9ca3af;
}

/* Line Edit & Text Inputs */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1f2937;
    color: #ffffff;
    border: 1px solid #4b5563;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #3b82f6;
}

QLineEdit:focus, QSpinBox:focus {
    border: 1px solid #3b82f6;
}

/* ComboBox */
QComboBox {
    background-color: #1f2937;
    color: #ffffff;
    border: 1px solid #4b5563;
    border-radius: 4px;
    padding: 4px;
    min-width: 6em;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 15px;
    border-left-width: 0px;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}

QComboBox QAbstractItemView {
    background-color: #1f2937;
    color: #ffffff;
    selection-background-color: #3b82f6;
    border: 1px solid #4b5563;
}

/* Lists & Trees */
QListWidget, QTreeWidget, QTableWidget {
    background-color: #1f2937;
    alternate-background-color: #273549;
    border: 1px solid #4b5563;
    border-radius: 4px;
    color: #ffffff;
}

QListWidget::item:selected, QTreeWidget::item:selected {
    background-color: #3b82f6;
    color: white;
}

QListWidget::item:hover, QTreeWidget::item:hover {
    background-color: #374151;
}

/* Sliders */
QSlider{
 background-color: none;
}

QSlider::groove:horizontal {
    border: 1px solid #4b5563;
    height: 4px;
    background: #4b5563;
    margin: 2px 0;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #9ca3af;
    border: 1px solid #9ca3af;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}

QSlider::add-page:vertical {
  background: none;
  border: none;
}

QRangeSlider {
      qproperty-barColor: #3b82f6;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #ffffff;
}

QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 2px;
}

/* QRangeSlider (SuperQt) */
QRangeSlider {
    qproperty-barColor: #3b82f6;
}

QRangeSlider::groove:horizontal {
    border: 1px solid #4b5563;
    height: 4px;
    background: #4b5563;
    margin: 2px 0;
    border-radius: 2px;
}

QRangeSlider::handle:horizontal {
    background: #9ca3af;
    border: 1px solid #9ca3af;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}

/* Menu Bar */
QMenuBar {
    background-color: #1f2937;
    color: #ffffff;
    border-bottom: 1px solid #374151;
}

QMenuBar::item {
    spacing: 3px;
    padding: 4px 8px;
    background: transparent;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #374151;
}

QMenu {
    background-color: #1f2937;
    color: #ffffff;
    border: 1px solid #4b5563;
}

QMenu::item {
    padding: 4px 24px 4px 8px;
}

QMenu::item:selected {
    background-color: #3b82f6;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #2b2b2b;
    width: 10px;
    margin: 0px 0px 0px 0px;
}

QScrollBar::handle:vertical {
    background: #4b5563;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #2b2b2b;
    height: 10px;
    margin: 0px 0px 0px 0px;
}

QScrollBar::handle:horizontal {
    background: #4b5563;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Labels */
QLabel {
    color: #e5e7eb;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #4b5563;
    border-radius: 4px;
    background-color: #2b2b2b;
    top: -1px;
}

QTabBar::tab {
    background-color: #374151;
    color: #9ca3af;
    border: 1px solid #4b5563;
    border-bottom: none;
    padding: 5px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #2b2b2b;
    color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #4b5563;
    color: #e5e7eb;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #4b5563;
    border-radius: 4px;
    margin-top: 1em;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
    color: #9ca3af;
}
"""
