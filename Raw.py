import cv2
import pytesseract
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np
from rapidfuzz import fuzz, process
from gpiozero import Button, LED

# === CONFIG ===
TESSERACT_CMD = "/usr/bin/tesseract"
LANGUAGE = "tha"
PLATE_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
CONFIDENCE_THRESHOLD = 0.5
BUTTON_GPIO = 2
green_led = LED(17)
red_led = LED(27)

# === INIT ===
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)

with open("/home/pi/Desktop/LicensePlate-EdgeAI/thai_provinces.txt", encoding="utf-8") as f:
    thai_provinces = [line.strip() for line in f.readlines()]
with open("/home/pi/Desktop/LicensePlate-EdgeAI/CarList.txt", encoding="utf-8") as f:
    car_list = [line.strip() for line in f.readlines()]

# === UTILS ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def pi_capture():
    with Picamera2() as cam:
        config = cam.create_still_configuration(main={"size": (1280, 720)})
        cam.configure(config)
        cam.start()
        time.sleep(2)
        frame = cam.capture_array()
    return frame

def plate_detectionandcrop(img):
    results = plate_model(img, conf=CONFIDENCE_THRESHOLD)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return safe_crop(img, x1, y1, x2, y2)
    return None

def seperate_part_and_textOCR(cropped_img):
    results = codeprov_model(cropped_img, conf=CONFIDENCE_THRESHOLD)[0]
    code_part, province_part = None, None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        part_img = safe_crop(cropped_img, x1, y1, x2, y2)

        if cls_id == 0:
            code_part = part_img
            cv2.imwrite("code_part.jpg", code_part)
        elif cls_id == 1:
            province_part = part_img
            cv2.imwrite("province_part.jpg", province_part)

    code_text = pytesseract.image_to_string(code_part, lang=LANGUAGE, config='--psm 7').strip() if code_part is not None else ''
    province_text = pytesseract.image_to_string(province_part, lang=LANGUAGE, config='--psm 7').strip() if province_part is not None else ''

    print(f"OCR Code: {code_text}")
    print(f"OCR Province: {province_text}")

    return code_text, province_text

def match_province(input_text, threshold=30):
    best_match = process.extractOne(input_text, thai_provinces, scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return ""

# === MAIN WORKFLOW ===
def on_button_pressed():
    print("🔍 Capturing image...")
    img = pi_capture()
    if img is None:
        print("❌ ไม่พบภาพจากกล้อง")
        return

    cropped_img = plate_detectionandcrop(img)
    if cropped_img is None:
        print("❌ ไม่พบป้ายทะเบียน")
        return

    code, prov_text = seperate_part_and_textOCR(cropped_img)
    province = match_province(prov_text)
    plate_txt = code.replace(" ", "") + province.replace(" ", "")
    print(f"📸 Plate: {plate_txt}")

    # เช็ค whitelist
    if plate_txt.strip() in car_list:
        green_led.on()
        red_led.off()
        print("✅ อนุญาตให้ผ่าน")
    else:
        green_led.off()
        red_led.on()
        print("❌ ไม่พบในระบบ")

    cv2.imwrite("detected_plate.jpg", cropped_img)

# === BUTTON SETUP ===
button = Button(BUTTON_GPIO)
button.when_pressed = on_button_pressed

print("📦 System ready. กดปุ่มเพื่อเริ่มตรวจจับ...")
from signal import pause
pause()
