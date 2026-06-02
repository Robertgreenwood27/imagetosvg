"""
Desktop UI for the photo-to-SVG engine.

Drop-in replacement for ui/desktop.py.

What this version improves:
- Darker, cleaner "vector lab" layout.
- Honest SVG preview that preserves aspect ratio.
- Clearer slider labels and descriptions.
- Visible busy/progress state while quantizing/tracing/smoothing.
- Presets for common output goals.
- Same engine calls as before: quantize -> cleanup_specks -> trace -> smooth_pass.
- Optional border-connected white background removal (before quantize).

Run with:    python -m ui.desktop
Or:          python ui/desktop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import (
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    QSize,
    QByteArray,
    QRectF,
)
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QKeySequence,
    QShortcut,
    QPainter,
    QColor,
    QFont,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFrame,
    QSizePolicy,
    QStatusBar,
    QSpinBox,
    QGroupBox,
    QProgressBar,
    QComboBox,
    QCheckBox,
)

# Make the engine importable whether we run as module or as script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from engine.quantize import quantize, cleanup_specks       # noqa: E402
from engine.trace import trace, TraceConfig                # noqa: E402
from engine.smooth import smooth_pass, SmoothConfig        # noqa: E402
from engine.bg_remove import remove_white_background, BgRemoveConfig  # noqa: E402


APP_QSS = """
QMainWindow {
    background: #0b0f14;
}

QWidget {
    background: #0b0f14;
    color: #d6dde8;
    font-size: 13px;
}

QGroupBox {
    background: #111821;
    border: 1px solid #263241;
    border-radius: 14px;
    margin-top: 16px;
    padding: 14px;
    font-weight: 700;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #cbd7e6;
}

QLabel {
    color: #d6dde8;
}

QPushButton {
    background: #1a2532;
    color: #e8eef8;
    border: 1px solid #344357;
    border-radius: 10px;
    padding: 8px 12px;
}

QPushButton:hover {
    background: #223043;
    border-color: #4b6380;
}

QPushButton:pressed {
    background: #172131;
}

QPushButton:disabled {
    background: #131922;
    color: #697584;
    border-color: #202a38;
}

QPushButton#PrimaryButton {
    background: #13a86b;
    color: #06110c;
    font-weight: 800;
    border: 1px solid #1fd98c;
}

QPushButton#PrimaryButton:hover {
    background: #19bf7c;
}

QPushButton#PassButton:checked {
    background: #13a86b;
    color: #06110c;
    font-weight: 800;
    border: 1px solid #1fd98c;
}

QComboBox, QSpinBox {
    background: #0d131b;
    color: #e8eef8;
    border: 1px solid #344357;
    border-radius: 8px;
    padding: 6px;
}

QSlider::groove:horizontal {
    height: 8px;
    background: #263241;
    border-radius: 4px;
}

QSlider::sub-page:horizontal {
    background: #13a86b;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #e8eef8;
    border: 2px solid #13a86b;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QProgressBar {
    background: #0d131b;
    color: #d6dde8;
    border: 1px solid #263241;
    border-radius: 8px;
    text-align: center;
    height: 18px;
}

QProgressBar::chunk {
    background: #13a86b;
    border-radius: 8px;
}

QStatusBar {
    background: #080b10;
    color: #96a4b7;
}

QCheckBox {
    color: #d6dde8;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #344357;
    border-radius: 5px;
    background: #0d131b;
}

QCheckBox::indicator:checked {
    background: #13a86b;
    border-color: #1fd98c;
    image: none;
}

QCheckBox::indicator:checked:hover {
    background: #19bf7c;
}

QCheckBox:disabled {
    color: #697584;
}

QCheckBox::indicator:disabled {
    border-color: #202a38;
    background: #131922;
}
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL image to a QPixmap, always via RGBA for safety."""
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def describe_colors(k: int) -> str:
    if k <= 4:
        return "Very graphic. Strong poster/sticker look, but detail will be sacrificed."
    if k <= 8:
        return "Clean graphic look. Good for decals, stickers, simple merch, and bold shapes."
    if k <= 14:
        return "Balanced detail. More image information while still staying printable/vector-friendly."
    return "High detail. Richer image, but more shapes and a heavier SVG."


def describe_speck(px: int) -> str:
    if px == 0:
        return "No cleanup. Keeps tiny artifacts and small isolated regions."
    if px <= 8:
        return "Light cleanup. Removes small noise while preserving most detail."
    if px <= 24:
        return "Medium cleanup. Better for clean SVGs and screenprint-like output."
    return "Heavy cleanup. Removes tiny islands aggressively; may erase small features."


def describe_max_dim(px: int) -> str:
    if px <= 900:
        return "Fast and simple. Lower detail, smaller SVG."
    if px <= 1700:
        return "Balanced. Good default for quality without making monster SVGs."
    return "High resolution. More detail, slower processing, larger SVG."


def describe_passes(n: int) -> str:
    if n <= 2:
        return "Light smoothing. Keeps sharper vector edges."
    if n <= 5:
        return "Balanced smoothing. Good default for clean merch-style SVGs."
    return "Heavy smoothing. Softer shapes, but small details can melt together."


def describe_bg_tolerance(v: int) -> str:
    if v <= 10:
        return "Strict: removes only near-pure white. Safe for images with light colors."
    if v <= 30:
        return "Balanced: removes white and slightly off-white backgrounds."
    if v <= 60:
        return "Loose: also catches cream, beige, or yellowish paper backgrounds."
    return "Aggressive: removes very light backgrounds; may eat into pale subject areas."


# ----------------------------------------------------------------------------
# Preview widgets
# ----------------------------------------------------------------------------

class ImagePane(QLabel):
    """A QLabel that holds a pixmap and rescales it to fit, preserving aspect."""

    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(300, 240))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("ImagePane")

        self._title = title
        self._original: QPixmap | None = None
        self.setText(f"{title}\n(empty)")
        self.setStyleSheet("""
            QLabel#ImagePane {
                background: #0d131b;
                color: #667386;
                border: 1px solid #263241;
                border-radius: 16px;
                padding: 10px;
            }
        """)

    def set_pixmap(self, pm: QPixmap | None):
        self._original = pm
        if pm is None:
            self.setText(f"{self._title}\n(empty)")
            self.setPixmap(QPixmap())
        else:
            self.setText("")
            self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._original is not None:
            self._rescale()

    def _rescale(self):
        if self._original is None:
            return
        scaled = self._original.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class SvgPane(QWidget):
    """
    Custom SVG preview pane.

    This intentionally does not use QSvgWidget's default scaling behavior.
    It paints the SVG into a centered QRectF that preserves the SVG aspect ratio.
    That fixes the "SVG gets wider / stretched" UI problem.
    """

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(300, 240))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._renderer: QSvgRenderer | None = None
        self._svg_text: str | None = None
        self._message = "SVG Result\n(empty)"
        self.setStyleSheet("""
            QWidget {
                background: #0d131b;
                border: 1px solid #263241;
                border-radius: 16px;
            }
        """)

    def load_svg_text(self, text: str | None):
        self._svg_text = text

        if not text:
            self._renderer = None
            self._message = "SVG Result\n(empty)"
            self.update()
            return

        renderer = QSvgRenderer()
        ok = renderer.load(QByteArray(text.encode("utf-8")))
        if ok and renderer.isValid():
            self._renderer = renderer
            self._message = ""
        else:
            self._renderer = None
            self._message = "Could not render SVG preview"

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Panel background. We draw it here so rounded corners are consistent.
        painter.setPen(QColor("#263241"))
        painter.setBrush(QColor("#0d131b"))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 16, 16)

        if self._renderer is None or not self._renderer.isValid():
            painter.setPen(QColor("#667386"))
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._message,
            )
            return

        default_size = self._renderer.defaultSize()
        svg_w = max(1, default_size.width())
        svg_h = max(1, default_size.height())
        svg_aspect = svg_w / svg_h

        padding = 14
        available_w = max(1, self.width() - padding * 2)
        available_h = max(1, self.height() - padding * 2)
        widget_aspect = available_w / available_h

        if widget_aspect > svg_aspect:
            render_h = available_h
            render_w = render_h * svg_aspect
        else:
            render_w = available_w
            render_h = render_w / svg_aspect

        x = (self.width() - render_w) / 2
        y = (self.height() - render_h) / 2

        self._renderer.render(painter, QRectF(x, y, render_w, render_h))


class LabeledSlider(QWidget):
    """
    Slider with a name, value badge, and live helper text.
    """

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        tick_interval: int,
        description_fn,
        suffix: str = "",
    ):
        super().__init__()
        self.description_fn = description_fn
        self.suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 800; color: #e8eef8;")
        self.value_label = QLabel("")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setMinimumWidth(80)
        self.value_label.setStyleSheet("""
            QLabel {
                background: #0d131b;
                border: 1px solid #263241;
                border-radius: 8px;
                padding: 4px 8px;
                color: #9ee8c5;
                font-weight: 800;
            }
        """)
        header.addWidget(self.title_label)
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(tick_interval)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #8f9bae; font-size: 12px;")
        layout.addWidget(self.description_label)

        self._on_value_changed(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int):
        self.slider.setValue(value)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def _on_value_changed(self, value: int):
        self.value_label.setText(f"{value}{self.suffix}")
        self.description_label.setText(self.description_fn(value))
        self.valueChanged.emit(value)


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------

class TraceWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(object)  # emits (raw_svg, [pass1, pass2, ...])
    failed = pyqtSignal(str)

    def __init__(
        self,
        quantized: Image.Image,
        trace_cfg: TraceConfig,
        smooth_cfg: SmoothConfig,
        n_passes: int,
    ):
        super().__init__()
        self.quantized = quantized
        self.trace_cfg = trace_cfg
        self.smooth_cfg = smooth_cfg
        self.n_passes = n_passes

    def run(self):
        try:
            self.progress.emit("Tracing color regions into SVG paths…")
            raw_svg = trace(self.quantized, self.trace_cfg)
            self.progress.emit(f"Trace complete: {len(raw_svg) // 1024} KB. Smoothing paths…")

            passes = []
            current = raw_svg
            for i in range(self.n_passes):
                current = smooth_pass(current, self.smooth_cfg)
                passes.append(current)
                self.progress.emit(
                    f"Smoothing pass {i + 1}/{self.n_passes} complete "
                    f"({len(current) // 1024} KB)"
                )

            self.finished_ok.emit((raw_svg, passes))
        except Exception as e:
            self.failed.emit(f"Error during trace/smooth: {e!r}")


class QuantizeWorker(QThread):
    """Runs (optional bg removal +) quantize on the full-resolution image."""

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(object)  # PIL Image
    failed = pyqtSignal(str)

    def __init__(
        self,
        image: Image.Image,
        k: int,
        merge_threshold: float,
        speck_min_px: int,
        remove_bg: bool,
        bg_cfg: BgRemoveConfig,
    ):
        super().__init__()
        self.image = image
        self.k = k
        self.merge_threshold = merge_threshold
        self.speck_min_px = speck_min_px
        self.remove_bg = remove_bg
        self.bg_cfg = bg_cfg

    def run(self):
        try:
            image = self.image

            if self.remove_bg:
                self.progress.emit("Removing border-connected white background…")
                image = remove_white_background(image, self.bg_cfg)

            self.progress.emit("Reducing image to selected color palette…")
            quantized, _palette = quantize(
                image,
                k=self.k,
                merge_threshold=self.merge_threshold,
            )
            if self.speck_min_px > 0:
                self.progress.emit("Cleaning small specks and isolated regions…")
                quantized = cleanup_specks(quantized, min_region_px=self.speck_min_px)
            self.finished_ok.emit(quantized)
        except Exception as e:
            self.failed.emit(f"Error during quantize: {e!r}")


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------

class MainWindow(QMainWindow):

    PREVIEW_MAX_DIM = 420
    DEFAULT_MAX_DIM = 1500

    PRESETS = {
        "Sticker / Screenprint": {
            "k": 8,
            "speck": 16,
            "maxdim": 1400,
            "passes": 5,
            "hint": "Bold shapes, limited palette, clean merch-friendly output.",
        },
        "Poster Detail": {
            "k": 12,
            "speck": 8,
            "maxdim": 1700,
            "passes": 5,
            "hint": "More detail while still keeping a graphic poster feel.",
        },
        "Simple Logo-ish": {
            "k": 5,
            "speck": 28,
            "maxdim": 1000,
            "passes": 6,
            "hint": "Very clean, simplified shapes. Good for icons or decals.",
        },
        "Detailed SVG": {
            "k": 18,
            "speck": 4,
            "maxdim": 2200,
            "passes": 3,
            "hint": "Richer image detail, heavier SVG, less aggressively simplified.",
        },
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo → SVG Lab")
        self.resize(1380, 920)

        self.full_image: Image.Image | None = None
        self.preview_source: Image.Image | None = None
        self.image_path: Path | None = None

        self.preview_quantized: Image.Image | None = None
        self.full_quantized: Image.Image | None = None

        self.svg_passes: list[str] = []
        self.current_pass_idx: int = 0
        self.pass_buttons: list[QPushButton] = []

        self._quantize_worker: QuantizeWorker | None = None
        self._trace_worker: TraceWorker | None = None
        self._one_pass_worker: QThread | None = None
        self._busy: bool = False

        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(180)
        self._preview_debounce.timeout.connect(self._update_preview_quantize)

        self._build_ui()
        self._wire_shortcuts()
        self._refresh_action_states()

    # -- Build --------------------------------------------------------------

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 16, 18, 12)
        outer.setSpacing(14)

        # Top bar
        top = QHBoxLayout()
        top.setSpacing(10)

        self.title_label = QLabel("Photo → SVG Lab")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 900; color: #f4f7fb;")

        self.open_btn = QPushButton("Open Image…")
        self.open_btn.clicked.connect(self.on_open)

        self.save_btn = QPushButton("Save Current SVG…")
        self.save_btn.clicked.connect(self.on_save)

        top.addWidget(self.title_label)
        top.addSpacing(12)
        top.addWidget(self.open_btn)
        top.addWidget(self.save_btn)
        top.addStretch(1)

        outer.addLayout(top)

        self.path_label = QLabel("No image loaded.")
        self.path_label.setStyleSheet("color: #8f9bae;")
        outer.addWidget(self.path_label)

        # Main content: previews left, controls right
        main = QHBoxLayout()
        main.setSpacing(14)
        outer.addLayout(main, 1)

        preview_col = QVBoxLayout()
        preview_col.setSpacing(10)
        main.addLayout(preview_col, 1)

        pane_labels = QHBoxLayout()
        pane_labels.addWidget(self._pane_title("Original"))
        pane_labels.addWidget(self._pane_title("Quantized Preview"))
        pane_labels.addWidget(self._pane_title("SVG Result"))
        preview_col.addLayout(pane_labels)

        panes = QHBoxLayout()
        panes.setSpacing(12)
        self.original_pane = ImagePane("Original")
        self.quant_pane = ImagePane("Quantized Preview")
        self.svg_pane = SvgPane()
        panes.addWidget(self.original_pane, 1)
        panes.addWidget(self.quant_pane, 1)
        panes.addWidget(self.svg_pane, 1)
        preview_col.addLayout(panes, 1)

        # Processing status row
        status_row = QHBoxLayout()
        self.activity_label = QLabel("Ready")
        self.activity_label.setStyleSheet("color: #9ee8c5; font-weight: 800;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")
        self.progress.setFixedWidth(260)

        status_row.addWidget(self.activity_label)
        status_row.addStretch(1)
        status_row.addWidget(self.progress)
        preview_col.addLayout(status_row)

        # Controls side
        controls_col = QVBoxLayout()
        controls_col.setSpacing(10)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_col)
        controls_widget.setFixedWidth(390)
        main.addWidget(controls_widget)

        workflow_box = QGroupBox("Look")
        workflow = QVBoxLayout(workflow_box)
        workflow.setSpacing(10)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        self.preset_hint = QLabel("")
        self.preset_hint.setWordWrap(True)
        self.preset_hint.setStyleSheet("color: #8f9bae;")

        workflow.addWidget(QLabel("Preset"))
        workflow.addWidget(self.preset_combo)
        workflow.addWidget(self.preset_hint)

        controls_col.addWidget(workflow_box)

        sliders_box = QGroupBox("Controls")
        sliders = QVBoxLayout(sliders_box)
        sliders.setSpacing(8)

        self.k_control = LabeledSlider(
            "Color Count",
            2,
            64,
            8,
            2,
            describe_colors,
        )
        self.k_control.valueChanged.connect(self._schedule_preview)

        self.speck_control = LabeledSlider(
            "Cleanup",
            0,
            64,
            16,
            8,
            describe_speck,
            " px",
        )
        self.speck_control.valueChanged.connect(self._schedule_preview)

        self.maxdim_control = LabeledSlider(
            "Trace Resolution",
            500,
            3000,
            self.DEFAULT_MAX_DIM,
            500,
            describe_max_dim,
            " px",
        )

        sliders.addWidget(self.k_control)
        sliders.addWidget(self.speck_control)
        sliders.addWidget(self.maxdim_control)

        controls_col.addWidget(sliders_box)

        # ---- Background Removal box ------------------------------------
        bg_box = QGroupBox("Background Removal")
        bg_layout = QVBoxLayout(bg_box)
        bg_layout.setSpacing(8)

        self.bg_check = QCheckBox("Remove white/near-white background")
        self.bg_check.setToolTip(
            "Flood-fills from border edges and removes pixels that are white or near-white.\n"
            "Runs before quantization so the background color never enters the palette.\n"
            "Best for product photos, scans, and clip-art on white paper."
        )
        self.bg_check.stateChanged.connect(self._on_bg_toggle)
        bg_layout.addWidget(self.bg_check)

        self.bg_hint = QLabel(
            "Off — background will be included in the color palette."
        )
        self.bg_hint.setWordWrap(True)
        self.bg_hint.setStyleSheet("color: #8f9bae; font-size: 12px;")
        bg_layout.addWidget(self.bg_hint)

        self.bg_tolerance_control = LabeledSlider(
            "Sensitivity",
            5,
            80,
            30,
            5,
            describe_bg_tolerance,
            "",
        )
        self.bg_tolerance_control.setEnabled(False)
        self.bg_tolerance_control.valueChanged.connect(self._schedule_preview)
        bg_layout.addWidget(self.bg_tolerance_control)

        controls_col.addWidget(bg_box)

        advanced_box = QGroupBox("Advanced")
        advanced = QGridLayout(advanced_box)

        advanced.addWidget(QLabel("Smoothing passes"), 0, 0)
        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 10)
        self.passes_spin.setValue(5)
        self.passes_spin.valueChanged.connect(self._update_pass_hint)
        advanced.addWidget(self.passes_spin, 0, 1)

        self.pass_hint = QLabel("")
        self.pass_hint.setWordWrap(True)
        self.pass_hint.setStyleSheet("color: #8f9bae;")
        advanced.addWidget(self.pass_hint, 1, 0, 1, 2)

        controls_col.addWidget(advanced_box)

        self.trace_btn = QPushButton("Trace SVG")
        self.trace_btn.setObjectName("PrimaryButton")
        self.trace_btn.setMinimumHeight(44)
        self.trace_btn.clicked.connect(self.on_trace)
        controls_col.addWidget(self.trace_btn)

        result_box = QGroupBox("Result")
        result = QVBoxLayout(result_box)
        result.setSpacing(10)

        pass_row = QHBoxLayout()
        pass_row.addWidget(QLabel("Pass"))
        self.pass_bar = QHBoxLayout()
        pass_row.addLayout(self.pass_bar, 1)
        result.addLayout(pass_row)

        result_buttons = QHBoxLayout()
        self.smooth_more_btn = QPushButton("+ Smooth More")
        self.smooth_more_btn.clicked.connect(self.on_smooth_more)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset_passes)
        result_buttons.addWidget(self.smooth_more_btn)
        result_buttons.addWidget(self.reset_btn)
        result.addLayout(result_buttons)

        controls_col.addWidget(result_box)
        controls_col.addStretch(1)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Open an image to begin.")

        self._apply_preset(self.preset_combo.currentText())
        self._update_pass_hint()

    def _pane_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: 900; color: #cbd7e6;")
        return label

    def _wire_shortcuts(self):
        QShortcut(QKeySequence("Left"), self).activated.connect(
            lambda: self._cycle_pass(-1)
        )
        QShortcut(QKeySequence("Right"), self).activated.connect(
            lambda: self._cycle_pass(+1)
        )
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.on_open)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.on_save)

    # -- Open / Save --------------------------------------------------------

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            self._error(f"Could not open image: {e}")
            return

        self.image_path = Path(path)
        self.full_image = img
        self.preview_source = self._downscale(img, self.PREVIEW_MAX_DIM)

        self.original_pane.set_pixmap(pil_to_qpixmap(img))
        self.path_label.setText(f"{self.image_path.name}  •  {img.width}×{img.height}")

        self.svg_passes = []
        self.current_pass_idx = 0
        self.full_quantized = None
        self._rebuild_pass_buttons()
        self.svg_pane.load_svg_text(None)

        self.statusBar().showMessage(
            f"Loaded {img.width}×{img.height}. Adjust the look, then click Trace SVG."
        )
        self._set_activity("Previewing color reduction…", indeterminate=True)
        self._schedule_preview()
        self._refresh_action_states()

    def on_save(self):
        if not self.svg_passes:
            return

        default_name = (
            f"{self.image_path.stem}_pass{self.current_pass_idx}.svg"
            if self.image_path else "output.svg"
        )
        default_dir = str(self.image_path.parent) if self.image_path else ""

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SVG",
            str(Path(default_dir) / default_name),
            "SVG (*.svg)",
        )
        if not path:
            return

        Path(path).write_text(self.svg_passes[self.current_pass_idx], encoding="utf-8")
        self.statusBar().showMessage(f"Saved {path}", 4000)
        self._set_activity("Saved SVG", indeterminate=False)

    # -- Background removal toggle ------------------------------------------

    def _on_bg_toggle(self, state: int):
        enabled = bool(state)
        self.bg_tolerance_control.setEnabled(enabled)
        if enabled:
            self.bg_hint.setText(
                "On — border-connected white pixels will be made transparent before tracing."
            )
        else:
            self.bg_hint.setText(
                "Off — background will be included in the color palette."
            )
        self._schedule_preview()

    def _bg_config(self) -> BgRemoveConfig:
        return BgRemoveConfig(
            luma_threshold=255 - self.bg_tolerance_control.value(),
            channel_tolerance=self.bg_tolerance_control.value(),
            feather_px=1,
        )

    # -- Presets / controls -------------------------------------------------

    def _apply_preset(self, name: str):
        preset = self.PRESETS.get(name)
        if not preset:
            return

        self.preset_hint.setText(preset["hint"])
        self.k_control.setValue(preset["k"])
        self.speck_control.setValue(preset["speck"])
        self.maxdim_control.setValue(preset["maxdim"])
        self.passes_spin.setValue(preset["passes"])
        self._update_pass_hint()

    def _update_pass_hint(self):
        self.pass_hint.setText(describe_passes(self.passes_spin.value()))

    def _schedule_preview(self, *_args):
        if self.preview_source is None or self._busy:
            return
        self._set_activity("Updating preview…", indeterminate=True)
        self._preview_debounce.start()

    def _update_preview_quantize(self):
        """Quantize (and optionally bg-remove) the downscaled preview source."""
        if self.preview_source is None:
            return

        k = self.k_control.value()
        speck = self.speck_control.value()
        remove_bg = self.bg_check.isChecked()

        try:
            source = self.preview_source

            if remove_bg:
                source = remove_white_background(source, self._bg_config())

            quantized, _ = quantize(source, k=k, merge_threshold=0.02)

            if speck > 0 and self.full_image is not None:
                scale = (
                    self.preview_source.width * self.preview_source.height
                ) / max(1, self.full_image.width * self.full_image.height)
                preview_speck = max(2, int(round(speck * scale)))
                quantized = cleanup_specks(quantized, min_region_px=preview_speck)

            self.preview_quantized = quantized
            self.quant_pane.set_pixmap(pil_to_qpixmap(quantized))
            self._set_activity("Preview updated", indeterminate=False)
            bg_note = " (bg removed)" if remove_bg else ""
            self.statusBar().showMessage(
                f"Preview updated: {k} colors, cleanup {speck}px{bg_note}."
            )
        except Exception as e:
            self._set_activity("Preview failed", indeterminate=False)
            self.statusBar().showMessage(f"Preview error: {e}", 5000)

    # -- Full pipeline ------------------------------------------------------

    def on_trace(self):
        if self.full_image is None:
            return

        self._set_busy(True)
        self._set_activity("Quantizing full-quality image…", indeterminate=True)
        self.statusBar().showMessage("Quantizing full-quality image…")

        max_dim = self.maxdim_control.value()
        src = self._downscale(self.full_image, max_dim)

        self._quantize_worker = QuantizeWorker(
            image=src,
            k=self.k_control.value(),
            merge_threshold=0.02,
            speck_min_px=self.speck_control.value(),
            remove_bg=self.bg_check.isChecked(),
            bg_cfg=self._bg_config(),
        )
        self._quantize_worker.progress.connect(
            lambda msg: (self.statusBar().showMessage(msg), self._set_activity(msg, True))
        )
        self._quantize_worker.finished_ok.connect(self._after_quantize)
        self._quantize_worker.failed.connect(self._on_worker_failed)
        self._quantize_worker.start()

    def _after_quantize(self, quantized: Image.Image):
        self.full_quantized = quantized
        self.quant_pane.set_pixmap(pil_to_qpixmap(quantized))

        n_passes = self.passes_spin.value()
        self._set_activity("Tracing SVG paths…", indeterminate=True)

        self._trace_worker = TraceWorker(
            quantized=quantized,
            trace_cfg=TraceConfig(),
            smooth_cfg=SmoothConfig(),
            n_passes=n_passes,
        )
        self._trace_worker.progress.connect(
            lambda msg: (self.statusBar().showMessage(msg), self._set_activity(msg, True))
        )
        self._trace_worker.finished_ok.connect(self._after_trace)
        self._trace_worker.failed.connect(self._on_worker_failed)
        self._trace_worker.start()

    def _after_trace(self, payload):
        raw_svg, passes = payload
        self.svg_passes = [raw_svg] + passes
        self.current_pass_idx = len(self.svg_passes) - 1

        self._rebuild_pass_buttons()
        self._show_current_pass()
        self._set_busy(False)

        self._set_activity("SVG ready", indeterminate=False)
        self.statusBar().showMessage(
            f"Done. Showing pass {self.current_pass_idx}. "
            "Use pass buttons or ←/→ to compare."
        )

    def on_smooth_more(self):
        if not self.svg_passes:
            return

        latest = self.svg_passes[-1]
        self._set_busy(True)
        self._set_activity("Adding one more smoothing pass…", indeterminate=True)
        self.statusBar().showMessage("Adding one more smoothing pass…")

        class _OnePassWorker(QThread):
            done = pyqtSignal(str)
            err = pyqtSignal(str)

            def __init__(self, svg: str, cfg: SmoothConfig):
                super().__init__()
                self.svg = svg
                self.cfg = cfg

            def run(self):
                try:
                    self.done.emit(smooth_pass(self.svg, self.cfg))
                except Exception as e:
                    self.err.emit(f"Error during extra smoothing pass: {e!r}")

        self._one_pass_worker = _OnePassWorker(latest, SmoothConfig())
        self._one_pass_worker.done.connect(self._after_one_more_pass)
        self._one_pass_worker.err.connect(self._on_worker_failed)
        self._one_pass_worker.start()

    def _after_one_more_pass(self, new_svg: str):
        self.svg_passes.append(new_svg)
        self.current_pass_idx = len(self.svg_passes) - 1

        self._rebuild_pass_buttons()
        self._show_current_pass()
        self._set_busy(False)

        self._set_activity("Extra pass ready", indeterminate=False)
        self.statusBar().showMessage(
            f"Now showing pass {self.current_pass_idx} ({len(new_svg) // 1024} KB)."
        )

    def on_reset_passes(self):
        self.svg_passes = []
        self.current_pass_idx = 0
        self._rebuild_pass_buttons()
        self.svg_pane.load_svg_text(None)
        self.statusBar().showMessage("Cleared SVG result. Click Trace SVG to start over.")
        self._set_activity("Result cleared", indeterminate=False)
        self._refresh_action_states()

    # -- Pass navigator -----------------------------------------------------

    def _rebuild_pass_buttons(self):
        for button in self.pass_buttons:
            button.deleteLater()
        self.pass_buttons = []

        while self.pass_bar.count():
            item = self.pass_bar.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i in range(len(self.svg_passes)):
            label = "Raw" if i == 0 else str(i)
            button = QPushButton(label)
            button.setObjectName("PassButton")
            button.setCheckable(True)
            button.setMinimumWidth(44)
            button.clicked.connect(lambda _checked, idx=i: self._select_pass(idx))
            self.pass_bar.addWidget(button)
            self.pass_buttons.append(button)

        self.pass_bar.addStretch(1)
        self._sync_pass_button_state()
        self._refresh_action_states()

    def _select_pass(self, idx: int):
        if not (0 <= idx < len(self.svg_passes)):
            return
        self.current_pass_idx = idx
        self._show_current_pass()
        self._sync_pass_button_state()

    def _cycle_pass(self, delta: int):
        if not self.svg_passes:
            return
        new_idx = max(0, min(len(self.svg_passes) - 1, self.current_pass_idx + delta))
        self._select_pass(new_idx)

    def _sync_pass_button_state(self):
        for i, button in enumerate(self.pass_buttons):
            button.setChecked(i == self.current_pass_idx)

    def _show_current_pass(self):
        if not self.svg_passes:
            self.svg_pane.load_svg_text(None)
            return

        svg = self.svg_passes[self.current_pass_idx]
        self.svg_pane.load_svg_text(svg)

        label = "Raw" if self.current_pass_idx == 0 else f"pass {self.current_pass_idx}"
        cubic_count = max(0, svg.count("C") - svg.count('"C'))

        self.statusBar().showMessage(
            f"Showing {label} — {len(svg) // 1024} KB, ~{cubic_count} cubic segments."
        )
        self._sync_pass_button_state()

    # -- State / misc -------------------------------------------------------

    def _set_activity(self, text: str, indeterminate: bool):
        self.activity_label.setText(text)

        if indeterminate:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Working…")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.progress.setFormat(text)

    def _set_busy(self, busy: bool):
        self._busy = busy

        widgets = (
            self.trace_btn,
            self.smooth_more_btn,
            self.k_control,
            self.speck_control,
            self.maxdim_control,
            self.passes_spin,
            self.open_btn,
            self.reset_btn,
            self.preset_combo,
            self.bg_check,
        )
        for widget in widgets:
            widget.setEnabled(not busy)

        # Keep the tolerance slider in sync with the checkbox state when not busy
        if not busy:
            self.bg_tolerance_control.setEnabled(self.bg_check.isChecked())
        else:
            self.bg_tolerance_control.setEnabled(False)

        if not busy:
            self._refresh_action_states()

    def _refresh_action_states(self):
        if self._busy:
            return

        has_image = self.full_image is not None
        has_passes = bool(self.svg_passes)

        self.trace_btn.setEnabled(has_image)
        self.save_btn.setEnabled(has_passes)
        self.smooth_more_btn.setEnabled(has_passes)
        self.reset_btn.setEnabled(has_passes)

    def _on_worker_failed(self, msg: str):
        self._set_busy(False)
        self._set_activity("Failed", indeterminate=False)
        self._error(msg)

    def _error(self, msg: str):
        self.statusBar().showMessage(msg, 8000)

    @staticmethod
    def _downscale(img: Image.Image, max_dim: int) -> Image.Image:
        if max(img.size) <= max_dim:
            return img

        scale = max_dim / max(img.size)
        new_size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale)),
        )
        return img.resize(new_size, Image.Resampling.LANCZOS)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())