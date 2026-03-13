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
MIN_RECORD_SECONDS = 0.35
LOW_INPUT_RMS_THRESHOLD = 0.00005
INPUT_DEVICE = os.getenv("DICTATE_INPUT_DEVICE")
PREFERRED_INPUT_NAME_HINTS = [
    "logitech webcam",
    "airpods",
    "macbook",
]

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


class RecordingOverlay:
    def __init__(self) -> None:
        self._state_queue = Queue()
        self._process = Process(target=self._run_overlay_process, args=(self._state_queue,), daemon=True)
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

    def close(self) -> None:
        if not self._process.is_alive():
            return
        try:
            self._state_queue.put_nowait("__quit__")
        except Exception:
            pass

    @staticmethod
    def _run_overlay_process(state_queue: Queue) -> None:
        if sys.platform == "darwin":
            try:
                RecordingOverlay._run_overlay_process_native(state_queue)
                return
            except Exception:
                pass

        try:
            RecordingOverlay._run_overlay_process_tk(state_queue)
        except Exception:
            return

    @staticmethod
    def _run_overlay_process_native(state_queue: Queue) -> None:
        import objc
        from AppKit import (
            NSApp,
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSBackingStoreBuffered,
            NSBezierPath,
            NSColor,
            NSGraphicsContext,
            NSScreen,
            NSScreenSaverWindowLevel,
            NSTimer,
            NSRoundLineCapStyle,
            NSView,
            NSWindow,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
            NSWindowStyleMaskBorderless,
        )
        from Foundation import NSMakePoint, NSMakeRect, NSObject

        window_width = 74.0
        window_height = 74.0
        top_margin = 0.0
        left_margin = 0.0

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

        def draw_round_rect(x: float, y: float, width: float, height: float, radius: float, fill_color, stroke_color=None, line_width: float = 0.0):
            if width <= 0 or height <= 0:
                return
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(x, y, width, height), radius, radius)
            fill_color.setFill()
            path.fill()
            if stroke_color is not None and line_width > 0:
                stroke_color.setStroke()
                path.setLineWidth_(line_width)
                path.stroke()

        def draw_line(x1: float, y1: float, x2: float, y2: float, stroke_color, line_width: float):
            path = NSBezierPath.bezierPath()
            path.setLineCapStyle_(NSRoundLineCapStyle)
            path.moveToPoint_(NSMakePoint(x1, y1))
            path.lineToPoint_(NSMakePoint(x2, y2))
            path.setLineWidth_(line_width)
            stroke_color.setStroke()
            path.stroke()

        def Jackdraw_clipboard_icon(cx: float, cy: float, scale: float, alpha: float):
            body_width = 14.0 * scale
            body_height = 16.0 * scale
            body_x = cx - body_width / 2.0
            body_y = cy - body_height / 2.0 - 0.8 * scale
            clip_width = 8.0 * scale
            clip_height = 4.3 * scale
            clip_x = cx - clip_width / 2.0
            clip_y = body_y + body_height - 1.5 * scale

            draw_round_rect(
                body_x,
                body_y,
                body_width,
                body_height,
                3.4 * scale,
                color("#d2d9e1", 0.90 * alpha),
                color("#f7fbff", 0.22 * alpha),
                1.0,
            )
            draw_round_rect(
                clip_x,
                clip_y,
                clip_width,
                clip_height,
                2.0 * scale,
                color("#b7c1cb", 0.94 * alpha),
                color("#edf2f7", 0.18 * alpha),
                0.8,
            )
            draw_line(
                body_x + 3.2 * scale,
                body_y + 10.8 * scale,
                body_x + body_width - 3.2 * scale,
                body_y + 10.8 * scale,
                color("#8f99a6", 0.72 * alpha),
                1.1 * scale,
            )
            draw_line(
                body_x + 3.2 * scale,
                body_y + 7.1 * scale,
                body_x + body_width - 4.4 * scale,
                body_y + 7.1 * scale,
                color("#a0aab6", 0.64 * alpha),
                1.0 * scale,
            )

        class OverlayView(NSView):
            def initWithFrame_(self, frame):
                self = objc.super(OverlayView, self).initWithFrame_(frame)
                if self is None:
                    return None
                self.state = "hidden"
                self.visibility = 0.0
                self.phase = 0.0
                self.idle_hold_frames = 0
                return self

            def isOpaque(self):
                return False

            def set_overlay_state(self, next_state: str) -> None:
                if next_state in {"recording", "transcribing"}:
                    self.state = next_state
                    self.idle_hold_frames = 0
                elif next_state == "saved":
                    self.state = "saved"
                    self.idle_hold_frames = 24
                else:
                    self.state = "idle"
                    self.idle_hold_frames = 0

            def step_animation(self) -> None:
                if self.state in {"recording", "transcribing"}:
                    target_visibility = 1.0
                elif self.state == "saved" and self.idle_hold_frames > 0:
                    self.idle_hold_frames -= 1
                    target_visibility = 1.0
                else:
                    target_visibility = 0.0
                    if self.visibility < 0.03:
                        self.state = "hidden"

                if target_visibility > self.visibility:
                    self.visibility += (target_visibility - self.visibility) * 0.20
                else:
                    self.visibility += (target_visibility - self.visibility) * 0.12

                self.visibility = max(0.0, min(1.0, self.visibility))
                self.phase += 0.10
                self.setNeedsDisplay_(True)

            def drawRect_(self, _dirty_rect):
                if self.state == "hidden" and self.visibility < 0.02:
                    return

                eased_visibility = 1.0 - pow(1.0 - self.visibility, 3)
                orb_scale = 0.16 + 0.84 * eased_visibility
                if self.state == "recording":
                    orb_scale *= 1.0 + 0.012 * math.sin(self.phase * 1.05)
                elif self.state == "transcribing":
                    orb_scale *= 1.0 + 0.006 * math.sin(self.phase * 0.85)
                elif self.state == "saved":
                    orb_scale *= 0.74

                alpha = min(0.93, pow(self.visibility, 0.86))
                center_x = window_width / 2.0
                center_y = window_height / 2.0
                aura_radius = 24.0 * orb_scale
                halo_radius = 19.5 * orb_scale
                shell_radius = 14.6 * orb_scale
                core_radius = 8.5 * orb_scale

                draw_circle(
                    center_x,
                    center_y,
                    aura_radius + 4.5 * orb_scale,
                    color("#020305", 0.14 * alpha),
                )
                draw_circle(
                    center_x,
                    center_y,
                    aura_radius,
                    color("#0a1016", 0.22 * alpha),
                )
                draw_circle(
                    center_x,
                    center_y,
                    halo_radius,
                    color("#1a222d", 0.52 * alpha),
                    color("#9ca7b5", 0.10 * alpha),
                    1.1,
                )
                draw_circle(
                    center_x,
                    center_y,
                    shell_radius,
                    color("#27313d", 0.48 * alpha),
                    color("#6f7a89", 0.10 * alpha),
                    0.9,
                )

                if self.state == "recording":
                    pulse = 0.5 + 0.5 * math.sin(self.phase * 1.8)
                    draw_circle(center_x, center_y, 12.4 * orb_scale + pulse * 0.8, color("#4c0d18", 0.22 * alpha))
                    draw_circle(center_x, center_y, 10.5 * orb_scale + pulse * 0.45, color("#701021", 0.20 * alpha))
                    draw_circle(
                        center_x,
                        center_y,
                        core_radius,
                        color("#ba2640", 0.94 * alpha),
                        color("#ee788a", 0.24 * alpha),
                        0.9,
                    )
                    draw_circle(
                        center_x - 2.7 * orb_scale,
                        center_y + 2.9 * orb_scale,
                        1.8 * orb_scale,
                        color("#ffd9df", 0.54 * alpha),
                    )
                    return

                if self.state == "transcribing":
                    draw_circle(center_x, center_y, 12.2 * orb_scale, color("#48111a", 0.22 * alpha))
                    orb_path = NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(center_x - core_radius, center_y - core_radius, core_radius * 2.0, core_radius * 2.0)
                    )
                    draw_circle(
                        center_x,
                        center_y,
                        core_radius,
                        color("#61121f", 0.82 * alpha),
                        color("#d65e72", 0.24 * alpha),
                        0.9,
                    )

                    context = NSGraphicsContext.currentContext()
                    context.saveGraphicsState()
                    orb_path.addClip()
                    offsets = (-5.4, -2.7, 0.0, 2.7, 5.4)
                    wave_colors = ("#cf5b6d", "#df7887", "#f6d8de", "#df7887", "#cf5b6d")
                    for index, offset in enumerate(offsets):
                        x_offset = offset * orb_scale + math.sin(self.phase * 1.25 - index * 0.8) * 0.42 * orb_scale
                        x_position = center_x + x_offset
                        half_limit = math.sqrt(max(core_radius * core_radius - x_offset * x_offset, 0.0)) * 0.82
                        activity = 0.48 + 0.34 * (0.5 + 0.5 * math.sin(self.phase * 1.6 + index * 0.55))
                        half_height = max(1.8 * orb_scale, half_limit * activity)
                        draw_line(
                            x_position,
                            center_y - half_height,
                            x_position,
                            center_y + half_height,
                            color(wave_colors[index], 0.88 * alpha),
                            max(1.6, 1.8 * orb_scale),
                        )
                    context.restoreGraphicsState()
                    return

                if self.state == "saved":
                    draw_circle(center_x, center_y, 10.8 * orb_scale, color("#111820", 0.24 * alpha))
                    draw_clipboard_icon(center_x, center_y, orb_scale, alpha)

        class OverlayController(NSObject):
            def initWithView_window_queue_(self, view, window, overlay_queue):
                self = objc.super(OverlayController, self).init()
                if self is None:
                    return None
                self.view = view
                self.window = window
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
                        self.window.orderOut_(None)
                        NSApp.terminate_(None)
                        return

                    self.view.set_overlay_state(next_state)

                self.view.step_animation()
                if self.view.state != "hidden" or self.view.visibility > 0.02:
                    self.window.orderFrontRegardless()

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        screens = NSScreen.screens()
        screen = screens[0] if screens else NSScreen.mainScreen()
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
        window.setIgnoresMouseEvents_(True)
        window.setReleasedWhenClosed_(False)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        view = OverlayView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, window_width, window_height))
        window.setContentView_(view)
        window.orderFrontRegardless()

        controller = OverlayController.alloc().initWithView_window_queue_(view, window, state_queue)
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
    def _run_overlay_process_tk(state_queue: Queue) -> None:
        import tkinter as tk

        window_size = 74
        center = window_size / 2
        bg_key = "#010203"

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.geometry(f"{window_size}x{window_size}+0+0")
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
            width=window_size,
            height=window_size,
            highlightthickness=0,
            bd=0,
            bg=bg_key,
        )
        canvas.pack()

        aura_id = canvas.create_oval(0, 0, 0, 0, fill="#0a1016", outline="", state="hidden")
        halo_id = canvas.create_oval(0, 0, 0, 0, fill="#1a222d", outline="#9ca7b5", width=1, state="hidden")
        shell_id = canvas.create_oval(0, 0, 0, 0, fill="#27313d", outline="#6f7a89", width=1, state="hidden")
        dot_id = canvas.create_oval(0, 0, 0, 0, fill="#ba2640", outline="#ee788a", width=1, state="hidden")
        clipboard_body_id = canvas.create_rectangle(0, 0, 0, 0, fill="#d2d9e1", outline="#f7fbff", width=1, state="hidden")
        clipboard_clip_id = canvas.create_rectangle(0, 0, 0, 0, fill="#b7c1cb", outline="#edf2f7", width=1, state="hidden")
        clipboard_line_ids = [
            canvas.create_line(0, 0, 0, 0, fill="#8f99a6", width=1, capstyle=tk.ROUND, state="hidden"),
            canvas.create_line(0, 0, 0, 0, fill="#a0aab6", width=1, capstyle=tk.ROUND, state="hidden"),
        ]

        state = "hidden"
        visibility = 0.0
        phase = 0.0
        idle_hold_frames = 0

        def set_circle(item_id, radius: float, cx: float = center, cy: float = center) -> None:
            canvas.coords(item_id, cx - radius, cy - radius, cx + radius, cy + radius)

        def tick() -> None:
            nonlocal state, visibility, phase, idle_hold_frames

            while True:
                try:
                    next_state = state_queue.get_nowait()
                except queue.Empty:
                    break
                if next_state == "__quit__":
                    root.destroy()
                    return
                if next_state in {"recording", "transcribing"}:
                    state = next_state
                    idle_hold_frames = 0
                elif next_state == "saved":
                    state = "saved"
                    idle_hold_frames = 24
                else:
                    state = "idle"
                    idle_hold_frames = 0

            if state in {"recording", "transcribing"}:
                target_visibility = 1.0
            elif state == "saved" and idle_hold_frames > 0:
                idle_hold_frames -= 1
                target_visibility = 1.0
            else:
                target_visibility = 0.0
                if visibility < 0.03:
                    state = "hidden"

            if target_visibility > visibility:
                visibility += (target_visibility - visibility) * 0.20
            else:
                visibility += (target_visibility - visibility) * 0.12
            visibility = max(0.0, min(1.0, visibility))
            phase += 0.10

            if state == "hidden" and visibility < 0.02:
                canvas.itemconfigure(aura_id, state="hidden")
                canvas.itemconfigure(halo_id, state="hidden")
                canvas.itemconfigure(shell_id, state="hidden")
                canvas.itemconfigure(dot_id, state="hidden")
                canvas.itemconfigure(clipboard_body_id, state="hidden")
                canvas.itemconfigure(clipboard_clip_id, state="hidden")
                for line_id in clipboard_line_ids:
                    canvas.itemconfigure(line_id, state="hidden")
                root.attributes("-alpha", 0.0)
                root.after(33, tick)
                return

            scale = 0.18 + 0.82 * (1.0 - pow(1.0 - visibility, 3))
            if state == "recording":
                scale *= 1.0 + 0.012 * math.sin(phase * 1.05)
            elif state == "transcribing":
                scale *= 1.0 + 0.006 * math.sin(phase * 0.85)
            elif state == "saved":
                scale *= 0.74

            set_circle(aura_id, 24.0 * scale)
            set_circle(halo_id, 19.5 * scale)
            set_circle(shell_id, 14.6 * scale)
            canvas.itemconfigure(halo_id, state="normal")
            canvas.itemconfigure(shell_id, state="normal")
            canvas.itemconfigure(aura_id, state="normal")
            canvas.itemconfigure(dot_id, state="hidden")
            canvas.itemconfigure(clipboard_body_id, state="hidden")
            canvas.itemconfigure(clipboard_clip_id, state="hidden")
            for line_id in clipboard_line_ids:
                canvas.itemconfigure(line_id, state="hidden")

            if state == "recording":
                set_circle(dot_id, 8.5 * scale)
                canvas.itemconfigure(dot_id, fill="#ba2640", outline="#ee788a", state="normal")
            elif state == "transcribing":
                set_circle(dot_id, 8.5 * scale)
                canvas.itemconfigure(dot_id, fill="#61121f", outline="#d65e72", state="normal")
            elif state == "saved":
                body_width = 14.0 * scale
                body_height = 16.0 * scale
                body_x1 = center - body_width / 2
                body_y1 = center - body_height / 2 - 0.8 * scale
                body_x2 = center + body_width / 2
                body_y2 = body_y1 + body_height
                clip_width = 8.0 * scale
                clip_height = 4.3 * scale
                clip_x1 = center - clip_width / 2
                clip_y1 = body_y2 - 1.5 * scale
                clip_x2 = center + clip_width / 2
                clip_y2 = clip_y1 + clip_height

                canvas.coords(clipboard_body_id, body_x1, body_y1, body_x2, body_y2)
                canvas.coords(clipboard_clip_id, clip_x1, clip_y1, clip_x2, clip_y2)
                canvas.itemconfigure(clipboard_body_id, state="normal")
                canvas.itemconfigure(clipboard_clip_id, state="normal")
                canvas.coords(
                    clipboard_line_ids[0],
                    body_x1 + 3.2 * scale,
                    body_y1 + 5.5 * scale,
                    body_x2 - 3.2 * scale,
                    body_y1 + 5.5 * scale,
                )
                canvas.coords(
                    clipboard_line_ids[1],
                    body_x1 + 3.2 * scale,
                    body_y1 + 9.1 * scale,
                    body_x2 - 4.2 * scale,
                    body_y1 + 9.1 * scale,
                )
                for line_id in clipboard_line_ids:
                    canvas.itemconfigure(line_id, state="normal")

            root.attributes("-alpha", round(min(0.93, pow(visibility, 0.86)), 3))
            root.after(33, tick)

        tick()
        root.mainloop()


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
        if overlay is not None:
            overlay.set_state("idle")
        print(f"Could not start audio input: {exc}")
        notify("Dictation", "Could not access microphone")
        return

    if overlay is not None:
        overlay.set_state("recording")
    play_chime("start")
    if overlay is None:
        notify("Recording...", "Speak now.")
    print("Recording started. Speak now...")


def paste_clipboard() -> None:
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script], check=False)


def stop_recording_and_transcribe() -> None:
    global is_recording, is_transcribing
    if not is_recording or is_transcribing:
        return

    saved_to_clipboard = False
    is_recording = False
    close_input_stream()
    is_transcribing = True
    if overlay is not None:
        overlay.set_state("transcribing")
    play_chime("stop")
    if overlay is None:
        notify("Dictation", "Transcribing")
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
            saved_to_clipboard = True
            print(f"\nTranscript:\n{text}\n")
            if overlay is None:
                notify("Dictation", "Copied to clipboard")

            if AUTO_PASTE:
                paste_clipboard()

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    finally:
        is_transcribing = False
        if overlay is not None:
            overlay.set_state("saved" if saved_to_clipboard else "idle")


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
            if is_recording:
                threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()
            else:
                start_recording()
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
    global overlay
    if not ensure_accessibility_permissions():
        return

    try:
        overlay = RecordingOverlay()
        if not overlay.is_available():
            overlay = None
    except Exception:
        overlay = None

    load_model()

    input_device = resolve_input_device()
    print(f"Input device: {describe_input_device(input_device)}")
    if INPUT_DEVICE:
        print(f"Device source: DICTATE_INPUT_DEVICE={INPUT_DEVICE}")

    print("Ready.")
    print("Start recording: hold Left Shift + Right Shift, then release both.")
    print("Stop recording: hold Left Shift + Right Shift, then release either Shift key.")
    print("Transcript is copied to clipboard and auto-pasted (Cmd+V).")
    if overlay is not None:
        print("Visual indicator: animated top-left recording badge.")
    else:
        print("Visual indicator unavailable: tkinter/_tkinter not present in this Python build.")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        close_input_stream()
        if overlay is not None:
            overlay.close()


if __name__ == "__main__":
    main()
