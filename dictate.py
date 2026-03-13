import os
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
audio_q = queue.Queue()
is_recording = False
is_transcribing = False
audio_chunks = []
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
        try:
            import tkinter as tk
        except Exception:
            return

        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.geometry("28x28+20+20")
            root.configure(bg="#000001")
            root.attributes("-alpha", 0.0)
            root.lift()
            root.resizable(False, False)

            try:
                root.call("tk::unsupported::MacWindowStyle", "style", root._w, "help", "none")
            except Exception:
                pass

            try:
                root.wm_attributes("-transparentcolor", "#00000101")
            except Exception:
                pass

            canvas = tk.Canvas(root, width=28, height=28, highlightthickness=0, bd=0, bg="#000001")
            canvas.pack()

            dot_id = canvas.create_oval(6, 6, 22, 22, fill="#ff2d2d", outline="#ff2d2d", state="hidden")
            ring_id = canvas.create_oval(5, 5, 23, 23, width=4, outline="#ff2d2d", state="hidden")

            state = "idle"
            rendered_state = None

            def apply_state(next_state: str) -> None:
                nonlocal rendered_state
                if next_state == rendered_state:
                    return

                if next_state == "recording":
                    canvas.itemconfigure(dot_id, state="normal")
                    canvas.itemconfigure(ring_id, state="hidden")
                    root.attributes("-alpha", 1.0)
                elif next_state == "transcribing":
                    canvas.itemconfigure(dot_id, state="hidden")
                    canvas.itemconfigure(ring_id, state="normal")
                    root.attributes("-alpha", 1.0)
                else:
                    canvas.itemconfigure(dot_id, state="hidden")
                    canvas.itemconfigure(ring_id, state="hidden")
                    root.attributes("-alpha", 0.0)

                rendered_state = next_state

            apply_state(state)

            def poll_state() -> None:
                nonlocal state
                while True:
                    try:
                        next_state = state_queue.get_nowait()
                    except queue.Empty:
                        break
                    if next_state == "__quit__":
                        root.destroy()
                        return
                    state = next_state
                apply_state(state)
                root.after(40, poll_state)

            poll_state()
            root.mainloop()
        except Exception:
            return


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
    audio_q.put(indata.copy())


def start_recording() -> None:
    global is_recording, audio_chunks
    if is_recording or is_transcribing:
        return
    audio_chunks = []
    is_recording = True
    if overlay is not None:
        overlay.set_state("recording")
    play_chime("start")
    title = "Recording..."
    message = "Speak now."
    notify(title, message)
    # print(title + " " + message)


def paste_clipboard() -> None:
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script], check=False)


def stop_recording_and_transcribe() -> None:
    global is_recording, is_transcribing
    if not is_recording or is_transcribing:
        return

    is_recording = False
    is_transcribing = True
    if overlay is not None:
        overlay.set_state("transcribing")
    play_chime("stop")
    notify("Dictation", "transcribing")
    print("Transcribing...")

    try:
        if not audio_chunks:
            print("No audio captured.")
            notify("Dictation", "No audio captured")
            return

        audio = np.concatenate(audio_chunks, axis=0)
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
            overlay.set_state("idle")


def audio_collector() -> None:
    global active_sample_rate, active_input_device
    active_input_device = resolve_input_device()
    with sd.InputStream(
        device=active_input_device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    ) as stream:
        active_sample_rate = int(stream.samplerate)
        while True:
            data = audio_q.get()
            if is_recording:
                audio_chunks.append(data)


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

    t = threading.Thread(target=audio_collector, daemon=True)
    t.start()

    input_device = resolve_input_device()
    print(f"Input device: {describe_input_device(input_device)}")
    if INPUT_DEVICE:
        print(f"Device source: DICTATE_INPUT_DEVICE={INPUT_DEVICE}")

    print("Ready.")
    print("Start recording: hold Left Shift + Right Shift, then release both.")
    print("Stop recording: hold Left Shift + Right Shift, then release either Shift key.")
    print("Transcript is copied to clipboard and auto-pasted (Cmd+V).")
    if overlay is not None:
        print("Visual indicator: red dot while recording, red ring while transcribing.")
    else:
        print("Visual indicator unavailable: tkinter/_tkinter not present in this Python build.")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        if overlay is not None:
            overlay.close()


if __name__ == "__main__":
    main()
