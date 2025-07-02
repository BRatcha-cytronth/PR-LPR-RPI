import cv2
import pytesseract
import re
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np
from rapidfuzz import fuzz, process

# === CONFIG ===
TESSERACT_CMD = "/usr/bin/tesseract"
LANGUAGE = "tha+eng"
PLATE_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
CONFIDENCE_THRESHOLD = 0.5

# === INIT ===
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)

with open("/home/pi/Desktop/LicensePlate-EdgeAI/thai_provinces.txt", encoding="utf-8") as f:
    thai_provinces = [line.strip() for line in f.readlines()]
# === UTILS ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# === FUNCTION 1: pi_capture ===
def pi_capture(trigger=False):
    if not trigger:
        return None
    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (1280, 720)})
    cam.configure(config)
    cam.start()
    time.sleep(2)
    frame = cam.capture_array()
    cam.stop()
    return frame

# === FUNCTION 2: plate_detectionandcrop ===
def plate_detectionandcrop(img):
    results = plate_model(img, conf=CONFIDENCE_THRESHOLD)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return safe_crop(img, x1, y1, x2, y2)
    return None

# === FUNCTION 3: seperate_part_and_textOCR ===
def seperate_part_and_textOCR(cropped_img):
    results = codeprov_model(cropped_img, conf=CONFIDENCE_THRESHOLD)[0]
    code_part, province_part = None, None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        part_img = safe_crop(cropped_img, x1, y1, x2, y2)
        if cls_id == 0:
            code_part = preprocess_for_ocr(part_img)
        elif cls_id == 1:
            province_part = preprocess_for_ocr(part_img)

    code_text = pytesseract.image_to_string(code_part, lang=LANGUAGE, config='--psm 7') if code_part is not None else ''
    province_text = pytesseract.image_to_string(province_part, lang=LANGUAGE, config='--psm 7') if province_part is not None else ''
    
    return code_text, province_text

def match_province(input_text, threshold=75):
    best_match = process.extractOne(input_text, thai_provinces, scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0], best_match[1]
    return None, 0

# === MAIN PIPELINE ===
def main():
    print("Capturing image...")
    img = pi_capture(trigger=True)
    if img is None:
        print("No image captured.")
        return

    cropped_img = plate_detectionandcrop(img)
    if cropped_img is None:
        print("No license plate detected.")
        return

    code, province = seperate_part_and_textOCR(cropped_img)
    
    province = match_province(province)
    print(f"Detected plate: {code} {province}")

    # Optional: Save cropped image
    cv2.imwrite("detected_plate.jpg", cropped_img)

if __name__ == "__main__":
    main()
