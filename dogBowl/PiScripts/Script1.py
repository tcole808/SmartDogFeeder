#!/usr/bin/env python3

import os
import time
import subprocess
from datetime import datetime

import requests
import RPi.GPIO as GPIO
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
TRIGGER_PIN = 17
RESULT_PIN = 27

FULL_IMAGE_PATH = "/home/pi/capture_full.jpg"
CROPPED_IMAGE_PATH = "/home/pi/capture_crop.jpg"

SERVER_URL = "http://192.168.1.100:5000/predict"   # CHANGE THIS
DOG_CONFIDENCE_THRESHOLD = 0.65

TRIGGER_COOLDOWN_SEC = 10.0
RESULT_PULSE_SEC = 1.0

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
REQUEST_TIMEOUT = 20

# ------------------------------------------
# CROP REGION
# (left, top, right, bottom)
# Adjust these to your bowl / target area
# ------------------------------------------
CROP_BOX = (160, 120, 480, 420)

# ==========================================
# GPIO SETUP
# ==========================================
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(RESULT_PIN, GPIO.OUT)
GPIO.output(RESULT_PIN, GPIO.LOW)


def capture_image(output_path: str) -> None:
    cmd = [
        "libcamera-still",
        "-n",
        "-o", output_path,
        "--width", str(CAPTURE_WIDTH),
        "--height", str(CAPTURE_HEIGHT),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Camera capture failed: {result.stderr.strip()}")


def crop_image(input_path: str, output_path: str, crop_box: tuple[int, int, int, int]) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = Image.open(input_path).convert("RGB")
    width, height = image.size

    left, top, right, bottom = crop_box

    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"Invalid crop box {crop_box} for image size {(width, height)}"
        )

    cropped = image.crop(crop_box)
    cropped.save(output_path, format="JPEG", quality=90)


def send_image_to_server(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        files = {
            "image": ("capture_crop.jpg", f, "image/jpeg")
        }

        response = requests.post(
            SERVER_URL,
            files=files,
            timeout=REQUEST_TIMEOUT
        )

    response.raise_for_status()
    return response.json()


def pulse_result_pin() -> None:
    GPIO.output(RESULT_PIN, GPIO.HIGH)
    time.sleep(RESULT_PULSE_SEC)
    GPIO.output(RESULT_PIN, GPIO.LOW)


def process_trigger() -> None:
    print("\nTrigger received. Capturing full image...")
    capture_image(FULL_IMAGE_PATH)
    print(f"Full image saved to {FULL_IMAGE_PATH}")

    print("Cropping image to region of interest...")
    crop_image(FULL_IMAGE_PATH, CROPPED_IMAGE_PATH, CROP_BOX)
    print(f"Cropped image saved to {CROPPED_IMAGE_PATH}")

    print("Sending cropped image to server for inference...")
    result = send_image_to_server(CROPPED_IMAGE_PATH)

    predicted_label = str(result.get("predicted_label", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    dog_detected = bool(result.get("dog_detected", False))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] label={predicted_label}, "
        f"confidence={confidence:.4f}, dog_detected={dog_detected}"
    )

    if dog_detected and confidence >= DOG_CONFIDENCE_THRESHOLD:
        print("DOG DETECTED with confidence >= 0.65 -> pulsing RESULT_PIN")
        pulse_result_pin()
    else:
        print("No valid dog detection. Not triggering feeder.")


def main() -> None:
    print("Pi client ready.")
    print(f"Listening for Arduino trigger on GPIO {TRIGGER_PIN}...")
    print(f"Using crop box: {CROP_BOX}")
    print(f"Dog confidence threshold: {DOG_CONFIDENCE_THRESHOLD:.2f}")

    last_trigger_time = 0.0

    try:
        while True:
            if GPIO.input(TRIGGER_PIN) == GPIO.HIGH:
                now = time.time()

                if now - last_trigger_time >= TRIGGER_COOLDOWN_SEC:
                    last_trigger_time = now

                    try:
                        process_trigger()
                    except Exception as exc:
                        print(f"Error while processing trigger: {exc}")

                    while GPIO.input(TRIGGER_PIN) == GPIO.HIGH:
                        time.sleep(0.05)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        GPIO.output(RESULT_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()