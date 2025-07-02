import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
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

# def preprocess_for_ocr(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return cv2.threshold(cv2.equalizeHist(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
            #cv2.imshow("🟩 CODE", code_part)
            cv2.imwrite("code_part.jpg", code_part)
        elif cls_id == 1:
            province_part = part_img
            #cv2.imshow("🟦 PROVINCE", province_part)
            cv2.imwrite("province_part.jpg", province_part)

#     cv2.waitKey(10)  # แสดงภาพชั่วคราว

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
        status_value.config(text="❌ ไม่พบภาพจากกล้อง", fg='red')
        return

    cropped_img = plate_detectionandcrop(img)
    if cropped_img is None:
        status_value.config(text="❌ ไม่พบป้ายทะเบียน", fg='red')
        return

    code, prov_text = seperate_part_and_textOCR(cropped_img)
    province = match_province(prov_text)
    plate_txt = code.replace(" ", "") + province.replace(" ", "")
    print(plate_txt)

    # เช็ค whitelist
    if plate_txt.strip() in car_list:
        green_led.on()
        red_led.off()
        status_value.config(text="✅ อนุญาตให้ผ่าน", fg='green')
    else:
        green_led.off()
        red_led.on()
        status_value.config(text="❌ ไม่พบในระบบ", fg='red')

    # อัปเดต GUI
    plate_value.config(text=code.strip())
    province_value.config(text=province.strip())

    # อัปเดตภาพใน GUI
    cv2.imwrite("detected_plate.jpg", cropped_img)
    image = Image.open("detected_plate.jpg").resize((640, 480))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# === GUI INIT ===
root = tk.Tk()
root.configure(bg='black')
root.title("License Plate Recognition")

# ภาพหลัก
default_img = Image.new("RGB", (640, 480), color='black')
photo = ImageTk.PhotoImage(default_img)
image_label = Label(root, image=photo)
image_label.grid(row=0, column=0, rowspan=6)


# Label GUI
Label(root, text="เลขทะเบียน", fg='skyblue', bg='black', font=("Prompt", 20, "bold")).grid(row=0, column=1, sticky='w')
plate_value = Label(root, text="---", fg='white', bg='grey', font=("Prompt", 18))
plate_value.grid(row=1, column=1, sticky='w')

Label(root, text="จังหวัด", fg='skyblue', bg='black', font=("Prompt", 20, "bold")).grid(row=2, column=1, sticky='w')
province_value = Label(root, text="---", fg='white', bg='grey', font=("Prompt", 18))
province_value.grid(row=3, column=1, sticky='w')

Label(root, text="สถานะ", fg='red', bg='black', font=("Prompt", 20, "bold")).grid(row=4, column=1, sticky='w')
status_value = Label(root, text="---", fg='white', bg='grey', font=("Prompt", 18))
status_value.grid(row=5, column=1, sticky='w')

# === GPIO TRIGGER ===
button = Button(BUTTON_GPIO)
button.when_pressed = on_button_pressed

# === START GUI LOOP ===
root.mainloop()
