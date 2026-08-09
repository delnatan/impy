"""Design tokens for pyvistra's flat dark theme ("Graphite & Amber").

Single source of truth for color: pyvistra/theme.py builds the Qt
stylesheet from these, and any widget or vispy visual that needs a
color matching the UI (rather than user/data-controlled color) should
import it from here instead of hardcoding a hex literal.
"""

# Backgrounds, darkest to lightest.
BG_BASE = "#1a1a19"
BG_SURFACE = "#221f1c"
BG_ELEVATED = "#2a2622"
BG_ELEVATED_HOVER = "#332e28"

# Borders.
BORDER = "#3a352f"
BORDER_FOCUS = "#e0a030"  # == ACCENT; named separately for stylesheet clarity.

# Text.
TEXT_PRIMARY = "#f2efe9"
TEXT_SECONDARY = "#a89f92"
TEXT_DISABLED = "#5f584e"
TEXT_FAINT = "#847c70"

# Accent (amber) and the text color that sits legibly on top of it.
ACCENT = "#e0a030"
ACCENT_HOVER = "#c98a1f"
ACCENT_PRESSED = "#b07819"
ON_ACCENT = "#1a1a19"

# Semantic status colors — distinct from ACCENT so badges/warnings don't
# read as interactive controls.
SUCCESS = "#4ecf9a"
WARNING = "#e8823a"
DANGER = "#f0555a"

# Corner radius: flat/hard-edged, not rounded. Kept as one token so any
# future adjustment is a single-line change.
RADIUS = "2px"
