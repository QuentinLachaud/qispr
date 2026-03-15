import os
import math
import queue
import tempfile
import threading
import subprocess
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from pynput import keyboard

try:
    from ApplicationServices import AXIsProcessTrusted
except Exception:
    print("Could not import AXIsProcessTrusted. Accessibility permission checks will be skipped.")
    AXIsProcessTrusted = None

# =========================
# Config
# =========================
MODEL_SIZE = "small"
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE = "en"
AUTO_PASTE = True
ENABLE_SYSTEM_NOTIFICATIONS = False
MIN_RECORD_SECONDS = 0.35
LOW_INPUT_RMS_THRESHOLD = 0.00005
INPUT_DEVICE = os.getenv("DICTATE_INPUT_DEVICE")
PREFERRED_INPUT_NAME_HINTS = [
    "logitech webcam",
    "airpods",
    "macbook",
]
OVERLAY_WINDOW_WIDTH = 118.0
OVERLAY_WINDOW_HEIGHT = 24.0
OVERLAY_MARGIN_X = 14.0
OVERLAY_MARGIN_Y = 6.0
OVERLAY_LABEL_TEXT = "Qispr"
OVERLAY_LABEL_FONT_NAME = "Monaco"
OVERLAY_LABEL_FONT_SIZE = 11.0
OVERLAY_LABEL_X = 24.0
OVERLAY_LABEL_Y_OFFSET = -0.5
OVERLAY_DOT_CENTER_X = 12.0
OVERLAY_DOT_RADIUS = 4.4
OVERLAY_CORNER_RADIUS = 12.0
OVERLAY_BACKGROUND_FILL = "#000000"
OVERLAY_BACKGROUND_ALPHA = 0.94
OVERLAY_HOVER_ALPHA = 0.46
OVERLAY_STATE_STEP_SECONDS = 1.0 / 30.0
OVERLAY_READY_PILL_WIDTH = 67.0
OVERLAY_ACTIVE_PILL_WIDTH = 116.0
OVERLAY_PILL_EXPAND_SPEED = 26.0
OVERLAY_PILL_CONTRACT_SPEED = 26.0
OVERLAY_RECORDING_VISUAL_DELAY = 0.18
OVERLAY_RECORDING_INTRO_MIN_SCALE = 0.22
OVERLAY_RECORDING_INTRO_SPEED = 18.0
OVERLAY_RECORDING_SETTLE_AMPLITUDE = 0.22
OVERLAY_RECORDING_SETTLE_DAMPING = 8.5
OVERLAY_RECORDING_SETTLE_FREQUENCY = 18.0
OVERLAY_WAVE_START_X = 66.0
OVERLAY_WAVE_DOT_COUNT = 10
OVERLAY_WAVE_DOT_SPACING = 3.6
OVERLAY_WAVE_MIN_RADIUS = 0.9
OVERLAY_WAVE_MAX_RADIUS = 1.55
OVERLAY_WAVE_MAX_OFFSET = 3.1
OVERLAY_STYLES = {
    "ready": {
        "fill": "#30d158",
    },
    "recording": {
        "fill": "#ff453a",
    },
    "transcribing": {
        "fill": "#30d158",
    },
}

# =========================
# State
# =========================
is_recording = False
is_transcribing = False
audio_chunks = []
audio_lock = threading.Lock()
input_stream = None
input_stream_lock = threading.Lock()
model = None
active_sample_rate = SAMPLE_RATE
active_input_device = None
overlay = None
overlay_lock = threading.Lock()
overlay_event_thread = None


def overlay_style(state: str) -> dict[str, str]:
    return OVERLAY_STYLES.get(state, OVERLAY_STYLES["ready"])


def overlay_pill_width(state: str, state_elapsed: float) -> float:
    if state == "recording":
        progress = 1.0 - math.exp(-state_elapsed * OVERLAY_PILL_EXPAND_SPEED)
        return OVERLAY_READY_PILL_WIDTH + (OVERLAY_ACTIVE_PILL_WIDTH - OVERLAY_READY_PILL_WIDTH) * progress
    if state == "transcribing":
        progress = 1.0 - math.exp(-state_elapsed * OVERLAY_PILL_CONTRACT_SPEED)
        return OVERLAY_ACTIVE_PILL_WIDTH - (OVERLAY_ACTIVE_PILL_WIDTH - OVERLAY_READY_PILL_WIDTH) * progress
    return OVERLAY_READY_PILL_WIDTH


def overlay_recording_visual_elapsed(state_elapsed: float) -> float:
    return max(0.0, state_elapsed - OVERLAY_RECORDING_VISUAL_DELAY)


def overlay_wave_visible(state: str, state_elapsed: float) -> bool:
    if state != "recording":
        return False
    if overlay_recording_visual_elapsed(state_elapsed) <= 0.0:
        return False
    return overlay_pill_width(state, state_elapsed) >= OVERLAY_ACTIVE_PILL_WIDTH - 1.5


class RecordingOverlay:
    def __init__(self) -> None:
        self._state_queue = Queue()
        self._event_queue = Queue()
        self._process = Process(target=self._run_overlay_process, args=(self._state_queue, self._event_queue), daemon=True)
        self._process.start()

    def set_state(self, state: str) -> None:
        if not self._process.is_alive():
            return
        try:
            self._state_queue.put_nowait(state)
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._process.is_alive()

    def get_event(self, timeout: float = 0.0) -> str | None:
        if not self._process.is_alive():
            return None
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception:
            return None

    def close(self) -> None:
        if not self._process.is_alive():
            return
        try:
            self._state_queue.put_nowait("__quit__")
        except Exception:
            pass

    @staticmethod
    def _run_overlay_process(state_queue: Queue, event_queue: Queue) -> None:
        if sys.platform == "darwin":
            try:
                RecordingOverlay._run_overlay_process_native(state_queue, event_queue)
                return
            except Exception:
                pass

        try:
            RecordingOverlay._run_overlay_process_tk(state_queue, event_queue)
        except Exception:
            return

    @staticmethod
    def _run_overlay_process_native(state_queue: Queue, event_queue: Queue) -> None:
        import objc
        from AppKit import (
            NSApp,
            NSAttributedString,
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSBackingStoreBuffered,
            NSBezierPath,
            NSColor,
            NSFont,
            NSFontAttributeName,
            NSForegroundColorAttributeName,
            NSScreen,
            NSScreenSaverWindowLevel,
            NSTrackingActiveAlways,
            NSTrackingArea,
            NSTrackingInVisibleRect,
            NSTrackingMouseEnteredAndExited,
            NSTimer,
            NSView,
            NSWindow,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
            NSWindowStyleMaskBorderless,
        )
        from Foundation import NSMakePoint, NSMakeRect, NSObject

        window_width = OVERLAY_WINDOW_WIDTH
        window_height = OVERLAY_WINDOW_HEIGHT
        top_margin = OVERLAY_MARGIN_Y
        left_margin = OVERLAY_MARGIN_X
        label_font = NSFont.fontWithName_size_(OVERLAY_LABEL_FONT_NAME, OVERLAY_LABEL_FONT_SIZE)
        if label_font is None:
            label_font = NSFont.systemFontOfSize_(OVERLAY_LABEL_FONT_SIZE)

        def color(hex_value: str, alpha: float = 1.0):
            red = int(hex_value[1:3], 16) / 255.0
            green = int(hex_value[3:5], 16) / 255.0
            blue = int(hex_value[5:7], 16) / 255.0
            return NSColor.colorWithCalibratedRed_green_blue_alpha_(red, green, blue, alpha)

        def draw_circle(cx: float, cy: float, radius: float, fill_color, stroke_color=None, line_width: float = 0.0):
            if radius <= 0:
                return
            path = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2))
            fill_color.setFill()
            path.fill()
            if stroke_color is not None and line_width > 0:
                stroke_color.setStroke()
                path.setLineWidth_(line_width)
                path.stroke()

        def draw_round_rect(x: float, y: float, width: float, height: float, radius: float, fill_color) -> None:
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, width, height),
                radius,
                radius,
            )
            fill_color.setFill()
            path.fill()

        def draw_label(text: str, x: float, center_y: float, text_color) -> None:
            label = NSAttributedString.alloc().initWithString_attributes_(
                text,
                {
                    NSFontAttributeName: label_font,
                    NSForegroundColorAttributeName: text_color,
                },
            )
            label_size = label.size()
            label.drawAtPoint_(NSMakePoint(x, center_y - label_size.height / 2.0 + OVERLAY_LABEL_Y_OFFSET))

        def draw_wave(start_x: float, center_y: float, phase: float) -> None:
            for index in range(OVERLAY_WAVE_DOT_COUNT):
                envelope = 0.35 + 0.65 * math.sin((index + 1) / (OVERLAY_WAVE_DOT_COUNT + 1) * math.pi)
                travel = phase * 1.9 - index * 0.55
                lift = math.sin(travel) * OVERLAY_WAVE_MAX_OFFSET * envelope
                intensity = 0.28 + 0.46 * (0.5 + 0.5 * math.sin(travel + 0.9)) * envelope
                radius = OVERLAY_WAVE_MIN_RADIUS + (OVERLAY_WAVE_MAX_RADIUS - OVERLAY_WAVE_MIN_RADIUS) * intensity
                draw_circle(
                    start_x + index * OVERLAY_WAVE_DOT_SPACING,
                    center_y + lift,
                    radius,
                    color("#ffffff", intensity),
                )

        class OverlayView(NSView):
            def initWithFrame_(self, frame):
                self = objc.super(OverlayView, self).initWithFrame_(frame)
                if self is None:
                    return None
                self.state = "ready"
                self.state_elapsed = 0.0
                self.phase = 0.0
                self.is_hovered = False
                tracking_area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                    NSMakeRect(0.0, 0.0, frame.size.width, frame.size.height),
                    NSTrackingMouseEnteredAndExited | NSTrackingActiveAlways | NSTrackingInVisibleRect,
                    self,
                    None,
                )
                self.addTrackingArea_(tracking_area)
                return self

            def isOpaque(self):
                return False

            def acceptsFirstMouse_(self, _event):
                return True

            def set_overlay_state(self, next_state: str) -> None:
                if next_state in {"ready", "recording", "transcribing"}:
                    resolved_state = next_state
                else:
                    resolved_state = "ready"
                if resolved_state != self.state:
                    self.state = resolved_state
                    self.state_elapsed = 0.0

            def step_animation(self) -> None:
                self.state_elapsed += OVERLAY_STATE_STEP_SECONDS
                self.phase += 0.10
                self.setNeedsDisplay_(True)

            def mouseEntered_(self, _event):
                self.is_hovered = True
                self.setNeedsDisplay_(True)

            def mouseExited_(self, _event):
                self.is_hovered = False
                self.setNeedsDisplay_(True)

            def mouseDown_(self, _event):
                try:
                    event_queue.put_nowait("toggle")
                except Exception:
                    pass

            def drawRect_(self, _dirty_rect):
                pill_width = overlay_pill_width(self.state, self.state_elapsed)
                dot_scale = 1.0
                if self.state == "recording":
                    visual_elapsed = overlay_recording_visual_elapsed(self.state_elapsed)
                    intro_progress = 1.0 - math.exp(-visual_elapsed * OVERLAY_RECORDING_INTRO_SPEED)
                    intro_scale = OVERLAY_RECORDING_INTRO_MIN_SCALE + (1.0 - OVERLAY_RECORDING_INTRO_MIN_SCALE) * intro_progress
                    settle_scale = 1.0 + OVERLAY_RECORDING_SETTLE_AMPLITUDE * math.exp(
                        -visual_elapsed * OVERLAY_RECORDING_SETTLE_DAMPING
                    ) * math.sin(visual_elapsed * OVERLAY_RECORDING_SETTLE_FREQUENCY)
                    breath_scale = 1.0 + 0.020 * math.sin(self.phase * 1.05)
                    dot_scale *= intro_scale * settle_scale * breath_scale
                elif self.state == "transcribing":
                    dot_scale *= 1.0 + 0.012 * math.sin(self.phase * 0.75)
                else:
                    dot_scale *= 1.0 + 0.010 * math.sin(self.phase * 0.55)

                style = overlay_style(self.state)
                center_x = OVERLAY_DOT_CENTER_X
                center_y = window_height / 2.0
                dot_radius = OVERLAY_DOT_RADIUS * dot_scale
                background_alpha = OVERLAY_HOVER_ALPHA if self.is_hovered else OVERLAY_BACKGROUND_ALPHA

                draw_round_rect(
                    0.0,
                    0.0,
                    pill_width,
                    window_height,
                    OVERLAY_CORNER_RADIUS,
                    color(OVERLAY_BACKGROUND_FILL, background_alpha),
                )
                draw_circle(center_x, center_y, dot_radius, color(style["fill"], 0.98))
                draw_label(OVERLAY_LABEL_TEXT, OVERLAY_LABEL_X, center_y, color("#ffffff", 0.82))
                if overlay_wave_visible(self.state, self.state_elapsed):
                    draw_wave(OVERLAY_WAVE_START_X, center_y, self.phase)

        class OverlayController(NSObject):
            def initWithViews_windows_queue_(self, views, windows, overlay_queue):
                self = objc.super(OverlayController, self).init()
                if self is None:
                    return None
                self.views = views
                self.windows = windows
                self.overlay_queue = overlay_queue
                self.timer = None
                return self

            def tick_(self, _timer):
                while True:
                    try:
                        next_state = self.overlay_queue.get_nowait()
                    except queue.Empty:
                        break

                    if next_state == "__quit__":
                        if self.timer is not None:
                            self.timer.invalidate()
                        for window in self.windows:
                            window.orderOut_(None)
                        NSApp.terminate_(None)
                        return

                    for view in self.views:
                        view.set_overlay_state(next_state)

                for view in self.views:
                    view.step_animation()
                for window in self.windows:
                    window.orderFrontRegardless()

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        primary_screen = NSScreen.mainScreen()
        if primary_screen is None:
            screens = list(NSScreen.screens())[:1]
        else:
            screens = [primary_screen]
        windows = []
        views = []
        for screen in screens:
            if screen is None:
                continue
            visible_frame = screen.visibleFrame()
            frame = NSMakeRect(
                visible_frame.origin.x + left_margin,
                visible_frame.origin.y + visible_frame.size.height - window_height - top_margin,
                window_width,
                window_height,
            )

            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            window.setOpaque_(False)
            window.setBackgroundColor_(NSColor.clearColor())
            window.setHasShadow_(False)
            window.setLevel_(NSScreenSaverWindowLevel)
            window.setIgnoresMouseEvents_(False)
            window.setReleasedWhenClosed_(False)
            window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            view = OverlayView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, window_width, window_height))
            window.setContentView_(view)
            window.orderFrontRegardless()
            windows.append(window)
            views.append(view)

        controller = OverlayController.alloc().initWithViews_windows_queue_(views, windows, state_queue)
        timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 30.0,
            controller,
            "tick:",
            None,
            True,
        )
        controller.timer = timer
        app.run()

    @staticmethod
    def _run_overlay_process_tk(state_queue: Queue, event_queue: Queue) -> None:
        import tkinter as tk

        window_width = int(OVERLAY_WINDOW_WIDTH)
        window_height = int(OVERLAY_WINDOW_HEIGHT)
        center_y = window_height / 2
        bg_key = "#010203"

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.geometry(f"{window_width}x{window_height}+{int(OVERLAY_MARGIN_X)}+{int(OVERLAY_MARGIN_Y)}")
        root.configure(bg=bg_key)
        root.attributes("-alpha", 0.0)
        root.lift()
        root.resizable(False, False)

        try:
            root.call("tk::unsupported::MacWindowStyle", "style", root._w, "help", "none")
        except Exception:
            pass

        try:
            root.wm_attributes("-transparentcolor", bg_key)
        except Exception:
            pass

        canvas = tk.Canvas(
            root,
            width=window_width,
            height=window_height,
            highlightthickness=0,
            bd=0,
            bg=bg_key,
        )
        canvas.pack()

        pill_id = canvas.create_rectangle(
            0,
            0,
            window_width,
            window_height,
            fill=OVERLAY_BACKGROUND_FILL,
            outline="",
            state="hidden",
        )

        dot_id = canvas.create_oval(0, 0, 0, 0, fill="#ba2640", outline="#ee788a", width=1, state="hidden")
        label_id = canvas.create_text(
            OVERLAY_LABEL_X,
            center_y + OVERLAY_LABEL_Y_OFFSET,
            anchor="w",
            fill="#ffffff",
            font=(OVERLAY_LABEL_FONT_NAME, int(OVERLAY_LABEL_FONT_SIZE)),
            text=OVERLAY_LABEL_TEXT,
            state="hidden",
        )
        wave_dot_ids = [
            canvas.create_oval(0, 0, 0, 0, fill="#ffffff", outline="", state="hidden")
            for _ in range(OVERLAY_WAVE_DOT_COUNT)
        ]

        state = "ready"
        state_elapsed = 0.0
        phase = 0.0
        is_hovered = False

        def set_circle(item_id, radius: float, cx: float = OVERLAY_DOT_CENTER_X, cy: float = center_y) -> None:
            canvas.coords(item_id, cx - radius, cy - radius, cx + radius, cy + radius)

        def set_hover_state(next_hover_state: bool) -> None:
            nonlocal is_hovered
            is_hovered = next_hover_state

        def handle_click(_event) -> None:
            try:
                event_queue.put_nowait("toggle")
            except Exception:
                pass

        canvas.bind("<Enter>", lambda _event: set_hover_state(True))
        canvas.bind("<Leave>", lambda _event: set_hover_state(False))
        canvas.bind("<Button-1>", handle_click)

        def tick() -> None:
            nonlocal state, state_elapsed, phase

            while True:
                try:
                    next_state = state_queue.get_nowait()
                except queue.Empty:
                    break
                if next_state == "__quit__":
                    root.destroy()
                    return
                if next_state in {"ready", "recording", "transcribing"}:
                    resolved_state = next_state
                else:
                    resolved_state = "ready"
                if resolved_state != state:
                    state = resolved_state
                    state_elapsed = 0.0

            state_elapsed += OVERLAY_STATE_STEP_SECONDS
            phase += 0.10

            pill_width = overlay_pill_width(state, state_elapsed)
            dot_scale = 1.0
            if state == "recording":
                visual_elapsed = overlay_recording_visual_elapsed(state_elapsed)
                intro_progress = 1.0 - math.exp(-visual_elapsed * OVERLAY_RECORDING_INTRO_SPEED)
                intro_scale = OVERLAY_RECORDING_INTRO_MIN_SCALE + (1.0 - OVERLAY_RECORDING_INTRO_MIN_SCALE) * intro_progress
                settle_scale = 1.0 + OVERLAY_RECORDING_SETTLE_AMPLITUDE * math.exp(
                    -visual_elapsed * OVERLAY_RECORDING_SETTLE_DAMPING
                ) * math.sin(visual_elapsed * OVERLAY_RECORDING_SETTLE_FREQUENCY)
                breath_scale = 1.0 + 0.020 * math.sin(phase * 1.05)
                dot_scale *= intro_scale * settle_scale * breath_scale
            elif state == "transcribing":
                dot_scale *= 1.0 + 0.012 * math.sin(phase * 0.75)
            else:
                dot_scale *= 1.0 + 0.010 * math.sin(phase * 0.55)

            style = overlay_style(state)
            background_alpha = OVERLAY_HOVER_ALPHA if is_hovered else OVERLAY_BACKGROUND_ALPHA

            canvas.itemconfigure(pill_id, state="normal")
            canvas.coords(pill_id, 0, 0, pill_width, window_height)
            set_circle(dot_id, OVERLAY_DOT_RADIUS * dot_scale)
            canvas.itemconfigure(dot_id, fill=style["fill"], outline="", state="normal")
            canvas.itemconfigure(label_id, fill="#ffffff", state="normal")
            if overlay_wave_visible(state, state_elapsed):
                for index, wave_dot_id in enumerate(wave_dot_ids):
                    envelope = 0.35 + 0.65 * math.sin((index + 1) / (OVERLAY_WAVE_DOT_COUNT + 1) * math.pi)
                    travel = phase * 1.9 - index * 0.55
                    lift = math.sin(travel) * OVERLAY_WAVE_MAX_OFFSET * envelope
                    intensity = 0.28 + 0.46 * (0.5 + 0.5 * math.sin(travel + 0.9)) * envelope
                    radius = OVERLAY_WAVE_MIN_RADIUS + (OVERLAY_WAVE_MAX_RADIUS - OVERLAY_WAVE_MIN_RADIUS) * intensity
                    x = OVERLAY_WAVE_START_X + index * OVERLAY_WAVE_DOT_SPACING
                    y = center_y + lift
                    shade = int(170 + 70 * intensity)
                    canvas.coords(wave_dot_id, x - radius, y - radius, x + radius, y + radius)
                    canvas.itemconfigure(wave_dot_id, fill=f"#{shade:02x}{shade:02x}{shade:02x}", state="normal")
            else:
                for wave_dot_id in wave_dot_ids:
                    canvas.itemconfigure(wave_dot_id, state="hidden")

            root.attributes("-alpha", background_alpha)
            root.after(33, tick)

        tick()
        root.mainloop()


def ensure_overlay_running() -> None:
    global overlay, overlay_event_thread
    with overlay_lock:
        if overlay is not None and overlay.is_available():
            return
        try:
            candidate = RecordingOverlay()
        except Exception:
            overlay = None
            return
        overlay = candidate if candidate.is_available() else None
        if overlay is not None:
            overlay_event_thread = threading.Thread(target=bridge_overlay_events, args=(overlay,), daemon=True)
            overlay_event_thread.start()


def set_overlay_state(state: str) -> None:
    ensure_overlay_running()
    with overlay_lock:
        if overlay is None or not overlay.is_available():
            return
        overlay.set_state(state)


def bridge_overlay_events(active_overlay: RecordingOverlay) -> None:
    while active_overlay.is_available():
        event = active_overlay.get_event(timeout=0.25)
        if event != "toggle":
            continue
        try:
            toggle_recording()
        except Exception as exc:
            print(f"Error toggling recording: {exc}")


def resolve_input_device():
    if not INPUT_DEVICE:
        for preferred_name in PREFERRED_INPUT_NAME_HINTS:
            for index, device in enumerate(sd.query_devices()):
                if device.get("max_input_channels", 0) <= 0:
                    continue
                if preferred_name in device["name"].lower():
                    return index
        return sd.default.device[0]

    try:
        return int(INPUT_DEVICE)
    except ValueError:
        pass

    target = INPUT_DEVICE.strip().lower()
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) <= 0:
            continue
        if target in device["name"].lower():
            return index

    print(f"Input device '{INPUT_DEVICE}' not found. Falling back to default input device.")
    return sd.default.device[0]


def describe_input_device(device_index) -> str:
    try:
        info = sd.query_devices(device_index)
        return f"{device_index}: {info['name']}"
    except Exception:
        return str(device_index)

shift_l_down = False
shift_r_down = False
start_combo_armed = False


def applescript_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str) -> None:
    if not ENABLE_SYSTEM_NOTIFICATIONS:
        return
    try:
        safe_title = applescript_escape(title)
        safe_message = applescript_escape(message)
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{safe_message}" with title "{safe_title}"',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def show_dialog(title: str, message: str) -> None:
    try:
        safe_title = applescript_escape(title)
        safe_message = applescript_escape(message.replace("\n", "\\n"))
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display dialog "{safe_message}" with title "{safe_title}" buttons {{"OK"}} default button "OK"',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def play_chime(kind: str) -> None:
    if sys.platform != "darwin":
        return

    sound_map = {
        "start": "/System/Library/Sounds/Glass.aiff",
        "stop": "/System/Library/Sounds/Blow.aiff"
        # "Tink.aiff",
    }
    sound_file = sound_map.get(kind)
    if not sound_file:
        return

    try:
        subprocess.Popen(
            ["afplay", "-v", "0.15", sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def load_model() -> None:
    global model
    if model is None:
        print("Loading faster-whisper model...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("Model loaded.")


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    if not is_recording:
        return
    with audio_lock:
        audio_chunks.append(indata.copy())


def open_input_stream() -> None:
    global input_stream, active_sample_rate, active_input_device

    close_input_stream()
    device = resolve_input_device()
    stream = sd.InputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()

    with input_stream_lock:
        input_stream = stream
    active_input_device = device
    active_sample_rate = int(stream.samplerate)


def close_input_stream() -> None:
    global input_stream

    with input_stream_lock:
        stream = input_stream
        input_stream = None

    if stream is None:
        return

    try:
        stream.stop()
    except Exception:
        pass

    try:
        stream.close()
    except Exception:
        pass

    try:
        sd.stop(ignore_errors=True)
    except Exception:
        pass


def start_recording() -> None:
    global is_recording, audio_chunks
    if is_recording or is_transcribing:
        return

    with audio_lock:
        audio_chunks = []

    is_recording = True
    try:
        open_input_stream()
    except Exception as exc:
        is_recording = False
        close_input_stream()
        set_overlay_state("ready")
        print(f"Could not start audio input: {exc}")
        notify("Dictation", "Could not access microphone")
        return

    set_overlay_state("recording")
    play_chime("start")
    print("Recording started. Speak now...")


def paste_clipboard() -> None:
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script], check=False)


def stop_recording_and_transcribe() -> None:
    global is_recording, is_transcribing
    if not is_recording or is_transcribing:
        return

    is_recording = False
    close_input_stream()
    is_transcribing = True
    set_overlay_state("transcribing")
    play_chime("stop")
    print("Transcribing...")

    try:
        with audio_lock:
            recorded_chunks = list(audio_chunks)

        if not recorded_chunks:
            print("No audio captured.")
            notify("Dictation", "No audio captured")
            return

        audio = np.concatenate(recorded_chunks, axis=0)
        duration_seconds = len(audio) / active_sample_rate
        rms = float(np.sqrt(np.mean(np.square(audio))))

        if duration_seconds < MIN_RECORD_SECONDS:
            print(f"Recording too short ({duration_seconds:.2f}s). Hold recording slightly longer.")
            notify("Dictation", "Recording too short")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(tmp_path, audio, active_sample_rate)

        try:
            segments, info = model.transcribe(
                str(tmp_path),
                language=LANGUAGE,
                vad_filter=True,
                beam_size=1,
            )
            text = "".join(seg.text for seg in segments).strip()

            if not text:
                segments, info = model.transcribe(
                    str(tmp_path),
                    language=LANGUAGE,
                    vad_filter=False,
                    beam_size=5,
                )
                text = "".join(seg.text for seg in segments).strip()

            if not text:
                if rms < LOW_INPUT_RMS_THRESHOLD:
                    print("Audio input level is very low. Check your microphone/input device.")
                    notify("Dictation", "Low microphone input detected")
                print("No speech detected.")
                notify("Dictation", "No speech detected")
                return

            pyperclip.copy(text)
            print(f"\nTranscript:\n{text}\n")

            if AUTO_PASTE:
                paste_clipboard()

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    finally:
        is_transcribing = False
        set_overlay_state("ready")


def toggle_recording() -> None:
    if is_transcribing:
        return
    if is_recording:
        threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()
        return
    start_recording()


def is_left_shift(key) -> bool:
    return key in {keyboard.Key.shift, keyboard.Key.shift_l}


def is_right_shift(key) -> bool:
    return key == keyboard.Key.shift_r


def on_press(key):
    global shift_l_down, shift_r_down, start_combo_armed

    if is_left_shift(key):
        shift_l_down = True
    elif is_right_shift(key):
        shift_r_down = True

    if not start_combo_armed and shift_l_down and shift_r_down:
        start_combo_armed = True
        if is_recording:
            print("Left Shift + Right Shift detected. Release either Shift key to stop recording.")
        else:
            print("Left Shift + Right Shift detected. Release either Shift key to start recording.")


def on_release(key):
    global shift_l_down, shift_r_down, start_combo_armed

    if start_combo_armed and (is_left_shift(key) or is_right_shift(key)):
        start_combo_armed = False
        shift_l_down = False
        shift_r_down = False
        try:
            toggle_recording()
        except Exception as e:
            print(f"Error toggling recording: {e}")
        return

    if is_left_shift(key):
        shift_l_down = False
    elif is_right_shift(key):
        shift_r_down = False



def ensure_accessibility_permissions() -> bool:
    if sys.platform != "darwin" or AXIsProcessTrusted is None:
        return True

    if AXIsProcessTrusted():
        return True

    print("\nmacOS is blocking global keyboard capture for this process.")
    print("To fix it, grant permissions in:")
    print("- System Settings → Privacy & Security → Accessibility")
    print("- System Settings → Privacy & Security → Input Monitoring")
    print("Enable your terminal app (and Python interpreter if needed), then restart this script.")
    notify("Dictation", "Grant Accessibility/Input Monitoring and restart")
    show_dialog(
        "Dictation needs macOS permissions",
        "Enable Accessibility and Input Monitoring for the app that launches Dictation "
        "(Automator or Terminal) and for the Python interpreter if needed, then launch it again.",
    )
    return False


def main() -> None:
    if not ensure_accessibility_permissions():
        return

    ensure_overlay_running()
    set_overlay_state("ready")

    load_model()

    input_device = resolve_input_device()
    print(f"Input device: {describe_input_device(input_device)}")
    if INPUT_DEVICE:
        print(f"Device source: DICTATE_INPUT_DEVICE={INPUT_DEVICE}")

    print("Ready.")
    print("Start recording: hold Left Shift + Right Shift, then release both.")
    print("Stop recording: hold Left Shift + Right Shift, then release either Shift key.")
    print("Transcript is copied to clipboard and auto-pasted (Cmd+V).")
    if overlay is not None and overlay.is_available():
        print("Visual indicator: status pill on the main display.")
    else:
        print("Visual indicator unavailable.")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        close_input_stream()
        with overlay_lock:
            if overlay is not None:
                overlay.close()


if __name__ == "__main__":
    main()
