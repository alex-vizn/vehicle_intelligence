import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime

import board
import neopixel
from gpiozero import Button


# ---------------- CONFIG ----------------

BASE_DIR = Path("/home/kartpi/kart_project")
SESSIONS_DIR = BASE_DIR / "data" / "sessions"
SRC_DIR = BASE_DIR / "src"

BUTTON_GPIO = 17          # physical pin 11
LED_PIN = board.D10       # GPIO10 / physical pin 19
LED_COUNT = 21

VIDEO_WIDTH = "640"
VIDEO_HEIGHT = "480"
FPS = "30"

CAMERA_CMD = "rpicam-vid"
IMU_SCRIPT = SRC_DIR / "record_imu.py"


# ---------------- LED ----------------

pixels = neopixel.NeoPixel(
    LED_PIN,
    LED_COUNT,
    brightness=0.2,
    auto_write=False,
    pixel_order=neopixel.GRB
)

def set_led(color):
    pixels.fill(color)
    pixels.show()

def led_ready():
    set_led((0, 255, 0))      # green

def led_recording():
    set_led((255, 0, 0))      # red

def led_saving():
    set_led((128, 0, 255))    # purple

def led_error():
    set_led((255, 80, 0))     # orange

def led_off():
    set_led((0, 0, 0))

def blink_green(times=10, delay=0.25):
    for _ in range(times):
        set_led((0, 255, 0))
        time.sleep(delay)
        led_off()
        time.sleep(delay)


# ---------------- SESSION ----------------

def create_session():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in SESSIONS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("session_")
    )

    session_name = f"session_{len(existing) + 1:03d}"
    session_path = SESSIONS_DIR / session_name

    raw_dir = session_path / "raw"
    processed_dir = session_path / "processed"
    notes_dir = session_path / "notes"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir()
    notes_dir.mkdir()

    video_file = raw_dir / f"{session_name}_video.mp4"
    imu_file = raw_dir / f"{session_name}_imu.csv"
    meta_file = session_path / "meta.json"

    meta = {
        "session_id": session_name,
        "created_at": datetime.now().isoformat(),
        "session_start_time_ns": time.time_ns(),
        "session_start_time_unix": time.time(),
        "video_file": f"raw/{session_name}_video.mp4",
        "imu_file": f"raw/{session_name}_imu.csv",
        "fps": int(FPS),
        "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "camera_model": "Pi Camera",
        "imu_model": "BNO085",
        "notes": ""
    }

    meta_file.write_text(json.dumps(meta, indent=2))

    return session_path, video_file, imu_file, meta_file, meta


# ---------------- RECORDING ----------------

recording = False
video_proc = None
imu_proc = None
current_meta = None
current_meta_file = None

def start_recording():
    global recording, video_proc, imu_proc, current_meta, current_meta_file

    if recording:
        return

    try:
        session_path, video_file, imu_file, meta_file, meta = create_session()

        current_meta = meta
        current_meta_file = meta_file

        print(f"Starting {session_path.name}")

        if IMU_SCRIPT.exists():
            imu_proc = subprocess.Popen([
                "python3",
                str(IMU_SCRIPT),
                str(session_path)
            ])
            print("IMU recording started")
        else:
            imu_proc = None
            print("WARNING: record_imu.py not found. IMU not recording.")

        time.sleep(0.3)

        meta["video_start_time_ns"] = time.time_ns()
        meta_file.write_text(json.dumps(meta, indent=2))

        video_proc = subprocess.Popen([
            CAMERA_CMD,
            "-t", "0",
            "--width", VIDEO_WIDTH,
            "--height", VIDEO_HEIGHT,
            "--framerate", FPS,
            "-o", str(video_file)
        ])

        recording = True
        led_recording()
        print("Video recording started")

    except Exception as e:
        print(f"ERROR starting recording: {e}")
        led_error()
        recording = False


def stop_process(proc, name):
    if proc is None:
        return

    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        time.sleep(1)

    if proc.poll() is None:
        proc.terminate()
        time.sleep(1)

    if proc.poll() is None:
        proc.kill()

    proc.wait()
    print(f"{name} stopped")


def stop_recording():
    global recording, video_proc, imu_proc, current_meta, current_meta_file

    if not recording:
        return

    try:
        print("Stopping session...")
        led_saving()

        stop_process(video_proc, "Video")
        stop_process(imu_proc, "IMU")

        if current_meta is not None and current_meta_file is not None:
            current_meta["session_end_time_ns"] = time.time_ns()
            current_meta["session_end_time_unix"] = time.time()
            current_meta["ended_at"] = datetime.now().isoformat()
            current_meta_file.write_text(json.dumps(current_meta, indent=2))

        recording = False
        video_proc = None
        imu_proc = None
        current_meta = None
        current_meta_file = None

        led_ready()
        print("Session saved")

    except Exception as e:
        print(f"ERROR stopping recording: {e}")
        led_error()


# ---------------- BUTTON ACTIONS ----------------

press_started_at = None
shutdown_started = False

def safe_shutdown():
    global shutdown_started

    print("Long press detected: safe shutdown")
    shutdown_started = True

    if recording:
        stop_recording()

    blink_green(times=12, delay=0.2)
    led_off()

    subprocess.run(["sudo", "shutdown", "now"])

def on_button_pressed():
    global press_started_at
    press_started_at = time.time()


def on_button_released():
    global press_started_at, shutdown_started

    if press_started_at is None:
        return

    press_duration = time.time() - press_started_at
    press_started_at = None

    if shutdown_started:
        return

    if press_duration < 3:
        if recording:
            stop_recording()
        else:
            start_recording()


def on_button_held():
    global shutdown_started
    shutdown_started = True
    safe_shutdown()


# ---------------- MAIN ----------------

button = Button(BUTTON_GPIO, hold_time=3, bounce_time=0.1)

button.when_pressed = on_button_pressed
button.when_released = on_button_released
button.when_held = on_button_held

print("Button recorder ready")
blink_green(times=3, delay=0.2)
led_ready()

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Manual exit")

    if recording:
        stop_recording()

    led_off()
    print("Exited cleanly")
