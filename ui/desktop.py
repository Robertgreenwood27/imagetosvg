"""
Desktop UI for the photo-to-SVG engine.

Three preview panes (Original | Quantized | SVG Result) plus a controls
strip and a pass navigator. The quantized preview updates live as you drag
the colors slider (on a downscaled copy, so it's snappy even on 12 MP photos).
Trace and smoothing run on a background thread so the UI never freezes.

Run with:    python -m ui.desktop
Or:          python ui/desktop.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QSize, QByteArray,
)
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut, QAction
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QSizePolicy,
    QStatusBar, QSpinBox, QGroupBox, QButtonGroup,
)

# Make the engine importable whether we run as module or as script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from engine.quantize import quantize, cleanup_specks       # noqa: E402
from engine.trace import trace, TraceConfig                # noqa: E402
from engine.smooth import smooth_pass, SmoothConfig        # noqa: E402


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL image to a QPixmap (always via RGBA for safety)."""
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    # Copy so the underlying bytes survive after this function returns.
    return QPixmap.fromImage(qimg.copy())


class ImagePane(QLabel):
    """A QLabel that holds a pixmap and rescales it to fit, preserving aspect."""

    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(280, 220))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #1e1e1e; color: #888; }"
        )
        self._title = title
        self._original: QPixmap | None = None
        self.setText(f"{title}\n(empty)")

    def set_pixmap(self, pm: QPixmap | None):
        self._original = pm
        if pm is None:
            self.setText(f"{self._title}\n(empty)")
            self.setPixmap(QPixmap())
        else:
            self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
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


class SvgPane(QSvgWidget):
    """QSvgWidget with a panel frame and minimum size."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(280, 220))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1e1e1e;")

    def load_svg_text(self, text: str | None):
        if not text:
            self.load(QByteArray())
            return
        self.load(QByteArray(text.encode("utf-8")))


# ----------------------------------------------------------------------------
# Worker: runs trace + smoothing passes on a background thread
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
            self.progress.emit("Tracing to SVG…")
            raw_svg = trace(self.quantized, self.trace_cfg)
            self.progress.emit(f"Traced ({len(raw_svg) // 1024} KB). Smoothing…")

            passes = []
            current = raw_svg
            for i in range(self.n_passes):
                current = smooth_pass(current, self.smooth_cfg)
                passes.append(current)
                self.progress.emit(
                    f"Smoothed pass {i + 1}/{self.n_passes} "
                    f"({len(current) // 1024} KB)"
                )

            self.finished_ok.emit((raw_svg, passes))
        except Exception as e:
            self.failed.emit(f"Error during trace/smooth: {e!r}")


class QuantizeWorker(QThread):
    """Runs quantize on the full-resolution image, with optional speck cleanup."""

    finished_ok = pyqtSignal(object)  # PIL Image
    failed = pyqtSignal(str)

    def __init__(
        self,
        image: Image.Image,
        k: int,
        merge_threshold: float,
        speck_min_px: int,
    ):
        super().__init__()
        self.image = image
        self.k = k
        self.merge_threshold = merge_threshold
        self.speck_min_px = speck_min_px

    def run(self):
        try:
            quantized, _palette = quantize(
                self.image, k=self.k, merge_threshold=self.merge_threshold
            )
            if self.speck_min_px > 0:
                quantized = cleanup_specks(quantized, min_region_px=self.speck_min_px)
            self.finished_ok.emit(quantized)
        except Exception as e:
            self.failed.emit(f"Error during quantize: {e!r}")


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------

class MainWindow(QMainWindow):

    PREVIEW_MAX_DIM = 400        # for live quantize preview during slider drag
    DEFAULT_MAX_DIM = 1500       # for the full-quality trace path

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo → SVG  (engine v2)")
        self.resize(1280, 820)

        # State
        self.full_image: Image.Image | None = None      # full-res original
        self.preview_source: Image.Image | None = None  # downscaled for live preview
        self.image_path: Path | None = None

        self.preview_quantized: Image.Image | None = None  # live posterized preview
        self.full_quantized: Image.Image | None = None     # used for actual trace

        self.svg_passes: list[str] = []   # [raw, pass1, pass2, ...]
        self.current_pass_idx: int = 0
        self.pass_buttons: list[QPushButton] = []

        self._quantize_worker: QuantizeWorker | None = None
        self._trace_worker: TraceWorker | None = None

        # Debounce timer for the colors slider (so we don't quantize on every
        # pixel of slider drag — only after the user pauses).
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(150)  # ms
        self._preview_debounce.timeout.connect(self._update_preview_quantize)

        self._build_ui()
        self._wire_shortcuts()
        self._refresh_action_states()

    # -- Build --------------------------------------------------------------

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # -- Top bar
        top = QHBoxLayout()
        self.open_btn = QPushButton("Open Image…")
        self.open_btn.clicked.connect(self.on_open)
        self.path_label = QLabel("(no image loaded)")
        self.path_label.setStyleSheet("color: #888;")
        self.save_btn = QPushButton("Save Current SVG…")
        self.save_btn.clicked.connect(self.on_save)
        top.addWidget(self.open_btn)
        top.addWidget(self.path_label, 1)
        top.addWidget(self.save_btn)
        outer.addLayout(top)

        # -- Three preview panes
        panes = QHBoxLayout()
        self.original_pane = ImagePane("Original")
        self.quant_pane = ImagePane("Quantized (live preview)")
        self.svg_pane = SvgPane()
        panes.addWidget(self.original_pane, 1)
        panes.addWidget(self.quant_pane, 1)
        panes.addWidget(self.svg_pane, 1)
        outer.addLayout(panes, 1)

        # -- Controls strip
        ctrls_box = QGroupBox("Settings")
        ctrls = QGridLayout(ctrls_box)

        # Colors slider
        ctrls.addWidget(QLabel("Colors"), 0, 0)
        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setRange(2, 24)
        self.k_slider.setValue(6)
        self.k_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.k_slider.setTickInterval(2)
        self.k_value = QLabel("6")
        self.k_value.setMinimumWidth(32)
        self.k_slider.valueChanged.connect(self._on_k_changed)
        ctrls.addWidget(self.k_slider, 0, 1)
        ctrls.addWidget(self.k_value, 0, 2)

        # Speck slider
        ctrls.addWidget(QLabel("Speck (px)"), 1, 0)
        self.speck_slider = QSlider(Qt.Orientation.Horizontal)
        self.speck_slider.setRange(0, 64)
        self.speck_slider.setValue(8)
        self.speck_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speck_slider.setTickInterval(8)
        self.speck_value = QLabel("8")
        self.speck_value.setMinimumWidth(32)
        self.speck_slider.valueChanged.connect(
            lambda v: (self.speck_value.setText(str(v)), self._schedule_preview())
        )
        ctrls.addWidget(self.speck_slider, 1, 1)
        ctrls.addWidget(self.speck_value, 1, 2)

        # Max dim (no live preview impact — only affects the full pipeline run)
        ctrls.addWidget(QLabel("Max dim (px)"), 2, 0)
        self.maxdim_slider = QSlider(Qt.Orientation.Horizontal)
        self.maxdim_slider.setRange(500, 3000)
        self.maxdim_slider.setSingleStep(100)
        self.maxdim_slider.setValue(self.DEFAULT_MAX_DIM)
        self.maxdim_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.maxdim_slider.setTickInterval(500)
        self.maxdim_value = QLabel(str(self.DEFAULT_MAX_DIM))
        self.maxdim_value.setMinimumWidth(40)
        self.maxdim_slider.valueChanged.connect(
            lambda v: self.maxdim_value.setText(str(v))
        )
        ctrls.addWidget(self.maxdim_slider, 2, 1)
        ctrls.addWidget(self.maxdim_value, 2, 2)

        # Passes selector
        ctrls.addWidget(QLabel("Initial passes"), 3, 0)
        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 10)
        self.passes_spin.setValue(5)
        ctrls.addWidget(self.passes_spin, 3, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        # Trace button
        self.trace_btn = QPushButton("Trace && Smooth")
        self.trace_btn.setMinimumHeight(36)
        self.trace_btn.setStyleSheet(
            "QPushButton { background:#2d5fbf; color:white; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self.trace_btn.clicked.connect(self.on_trace)
        ctrls.addWidget(self.trace_btn, 0, 3, 4, 1)

        outer.addWidget(ctrls_box)

        # -- Pass navigator
        nav_box = QGroupBox("Result")
        nav = QHBoxLayout(nav_box)
        nav.addWidget(QLabel("Pass:"))
        self.pass_bar = QHBoxLayout()
        self.pass_bar_widget = QWidget()
        self.pass_bar_widget.setLayout(self.pass_bar)
        nav.addWidget(self.pass_bar_widget, 1)
        self.smooth_more_btn = QPushButton("+ Smooth More")
        self.smooth_more_btn.clicked.connect(self.on_smooth_more)
        nav.addWidget(self.smooth_more_btn)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset_passes)
        nav.addWidget(self.reset_btn)
        outer.addWidget(nav_box)

        # -- Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Open an image to begin.")

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
            img = Image.open(path).convert("RGB")
        except Exception as e:
            self._error(f"Could not open image: {e}")
            return

        self.image_path = Path(path)
        self.full_image = img
        # Build a downscaled preview source — quantization on a 12 MP photo
        # would lock the slider; on a 400 px wide thumbnail it's instant.
        self.preview_source = self._downscale(img, self.PREVIEW_MAX_DIM)

        self.original_pane.set_pixmap(pil_to_qpixmap(img))
        self.path_label.setText(f"{self.image_path.name}  ({img.width}×{img.height})")
        self.statusBar().showMessage(
            f"Loaded {img.width}×{img.height}. Adjust colors slider, then click Trace."
        )

        # Reset any prior results
        self.svg_passes = []
        self.current_pass_idx = 0
        self.full_quantized = None
        self._rebuild_pass_buttons()
        self.svg_pane.load_svg_text(None)

        # Kick off the live preview at the current slider value
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
            self, "Save SVG", str(Path(default_dir) / default_name),
            "SVG (*.svg)",
        )
        if not path:
            return
        Path(path).write_text(self.svg_passes[self.current_pass_idx], encoding="utf-8")
        self.statusBar().showMessage(f"Saved {path}", 4000)

    # -- Live preview quantization (debounced) ------------------------------

    def _on_k_changed(self, v: int):
        self.k_value.setText(str(v))
        self._schedule_preview()

    def _schedule_preview(self):
        if self.preview_source is None:
            return
        self._preview_debounce.start()

    def _update_preview_quantize(self):
        """Quantize the *downscaled* preview source. Fast even on 12 MP photos."""
        if self.preview_source is None:
            return
        k = self.k_slider.value()
        speck = self.speck_slider.value()
        try:
            quantized, _ = quantize(self.preview_source, k=k, merge_threshold=0.02)
            if speck > 0:
                # Scale speck threshold proportionally so the preview matches
                # what the full-res trace will see.
                scale = (self.preview_source.width * self.preview_source.height) / \
                        max(1, self.full_image.width * self.full_image.height)
                preview_speck = max(2, int(round(speck * scale)))
                quantized = cleanup_specks(quantized, min_region_px=preview_speck)
            self.preview_quantized = quantized
            self.quant_pane.set_pixmap(pil_to_qpixmap(quantized))
        except Exception as e:
            self.statusBar().showMessage(f"Preview error: {e}", 4000)

    # -- Full pipeline ------------------------------------------------------

    def on_trace(self):
        if self.full_image is None:
            return
        # Disable controls during work
        self._set_busy(True)
        self.statusBar().showMessage("Quantizing full-resolution image…")

        # Step 1: full-quality quantize (in a thread, on the downscaled-to-max-dim
        # full image). We don't reuse the live preview because that was at a
        # much smaller resolution.
        max_dim = self.maxdim_slider.value()
        src = (self.full_image if max_dim == 0
               else self._downscale(self.full_image, max_dim))

        self._quantize_worker = QuantizeWorker(
            image=src,
            k=self.k_slider.value(),
            merge_threshold=0.02,
            speck_min_px=self.speck_slider.value(),
        )
        self._quantize_worker.finished_ok.connect(self._after_quantize)
        self._quantize_worker.failed.connect(self._on_worker_failed)
        self._quantize_worker.start()

    def _after_quantize(self, quantized: Image.Image):
        self.full_quantized = quantized
        # Replace the preview pane with the full-quality quantized image,
        # since it's now the authoritative version.
        self.quant_pane.set_pixmap(pil_to_qpixmap(quantized))

        n_passes = self.passes_spin.value()
        self._trace_worker = TraceWorker(
            quantized=quantized,
            trace_cfg=TraceConfig(),
            smooth_cfg=SmoothConfig(),
            n_passes=n_passes,
        )
        self._trace_worker.progress.connect(self.statusBar().showMessage)
        self._trace_worker.finished_ok.connect(self._after_trace)
        self._trace_worker.failed.connect(self._on_worker_failed)
        self._trace_worker.start()

    def _after_trace(self, payload):
        raw_svg, passes = payload
        self.svg_passes = [raw_svg] + passes  # index 0 = raw
        # Default selection: the last pass (usually the smoothest)
        self.current_pass_idx = len(self.svg_passes) - 1
        self._rebuild_pass_buttons()
        self._show_current_pass()
        self._set_busy(False)
        self.statusBar().showMessage(
            f"Done. {len(passes)} smoothing passes available. "
            "Click pass buttons or use ←/→ to compare."
        )

    def on_smooth_more(self):
        if not self.svg_passes:
            return
        # Run one more smoothing pass on the latest in a thread.
        latest = self.svg_passes[-1]
        self._set_busy(True)
        self.statusBar().showMessage("Adding one more smoothing pass…")

        class _OnePassWorker(QThread):
            done = pyqtSignal(str)
            err = pyqtSignal(str)

            def __init__(self, svg, cfg):
                super().__init__()
                self.svg, self.cfg = svg, cfg

            def run(self):
                try:
                    self.done.emit(smooth_pass(self.svg, self.cfg))
                except Exception as e:
                    self.err.emit(repr(e))

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
        self.statusBar().showMessage(
            f"Now at pass {self.current_pass_idx} ({len(new_svg) // 1024} KB)."
        )

    def on_reset_passes(self):
        self.svg_passes = []
        self.current_pass_idx = 0
        self._rebuild_pass_buttons()
        self.svg_pane.load_svg_text(None)
        self.statusBar().showMessage("Cleared. Click Trace to start over.")
        self._refresh_action_states()

    # -- Pass navigator -----------------------------------------------------

    def _rebuild_pass_buttons(self):
        # Wipe old buttons
        for b in self.pass_buttons:
            b.deleteLater()
        self.pass_buttons = []
        while self.pass_bar.count():
            item = self.pass_bar.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i in range(len(self.svg_passes)):
            label = "Raw" if i == 0 else str(i)
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumWidth(46)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 10px; }"
                "QPushButton:checked { background: #2d5fbf; color: white; "
                "font-weight: bold; border: 1px solid #1a3d80; }"
            )
            btn.clicked.connect(lambda _checked, idx=i: self._select_pass(idx))
            self.pass_bar.addWidget(btn)
            self.pass_buttons.append(btn)
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
        for i, b in enumerate(self.pass_buttons):
            b.setChecked(i == self.current_pass_idx)

    def _show_current_pass(self):
        if not self.svg_passes:
            self.svg_pane.load_svg_text(None)
            return
        svg = self.svg_passes[self.current_pass_idx]
        self.svg_pane.load_svg_text(svg)
        label = "Raw" if self.current_pass_idx == 0 else f"pass {self.current_pass_idx}"
        n_cubics = svg.count("C") - svg.count('"C')  # crude but fine
        self.statusBar().showMessage(
            f"Showing {label} — {len(svg) // 1024} KB, ~{n_cubics} cubic segments"
        )

    # -- Misc ---------------------------------------------------------------

    def _set_busy(self, busy: bool):
        for w in (self.trace_btn, self.smooth_more_btn, self.k_slider,
                  self.speck_slider, self.maxdim_slider, self.passes_spin,
                  self.open_btn, self.reset_btn):
            w.setEnabled(not busy)
        if not busy:
            self._refresh_action_states()

    def _refresh_action_states(self):
        has_image = self.full_image is not None
        has_passes = bool(self.svg_passes)
        self.trace_btn.setEnabled(has_image)
        self.save_btn.setEnabled(has_passes)
        self.smooth_more_btn.setEnabled(has_passes)
        self.reset_btn.setEnabled(has_passes)

    def _on_worker_failed(self, msg: str):
        self._set_busy(False)
        self._error(msg)

    def _error(self, msg: str):
        self.statusBar().showMessage(msg, 8000)

    @staticmethod
    def _downscale(img: Image.Image, max_dim: int) -> Image.Image:
        if max(img.size) <= max_dim:
            return img
        scale = max_dim / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        return img.resize(new_size, Image.LANCZOS)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
