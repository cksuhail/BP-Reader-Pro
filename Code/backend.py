"""
This module implements a complete backend pipeline for reading digital blood
pressure monitor values from images.

It uses a YOLO object detection model to locate the BP display area, applies
multiple image preprocessing strategies, performs OCR using EasyOCR, detects
monitor error messages, and finally returns structured JSON containing
SYS/DIA/HR readings or error information.
"""

# =========================================================
# STANDARD IMPORTS
# =========================================================
import sys
import os
import warnings
import logging
from datetime import datetime


# =========================================================
# OUTPUT SUPPRESSION UTILITY
# =========================================================
# This context manager temporarily suppresses stdout/stderr.
# It is used to silence verbose logs from YOLO and EasyOCR.
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Disable warnings and YOLO logs globally
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# =========================================================
# SILENT THIRD-PARTY IMPORTS
# =========================================================
with SuppressOutput():
    from ultralytics import YOLO
    import easyocr
    import cv2
    import re
    import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

# Path to trained YOLO model for BP display detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "Models", "Models/BP_model_best .pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found at: {MODEL_PATH}\n"
        "Make sure Best_yolo_crop.pt exists inside the Models folder."
    )

# Minimum confidence for YOLO detections
CONF_THRES = 0.3

# OCR language configuration
LANGS = ['en']

# Enable GPU acceleration for EasyOCR if available
USE_GPU = True

# OCR noise tokens that commonly appear on BP monitors.
# These are stripped before extracting numeric values.
ALLOWED_UNITS = [
    'mmhg',   # Blood pressure unit
    '/min',   # Pulse per minute
    'min',
    'bpm',    # Beats per minute
    'hr',     # Heart rate label
    'sys',    # Systolic label
    'dia',    # Diastolic label
    'pul'     # Pulse label
]

# Known BP monitor error messages and corrective actions
KNOWN_ERRORS = [
    {
        "code": "ERROR 1",
        "keywords": ["tighter", "apply", "cuff", "error 1"],
        "solution": "Apply the arm cuff tighter."
    },
    {
        "code": "ERROR 2",
        "keywords": ["move", "talk", "still", "remain", "error 2"],
        "solution": "Do not move or talk, remain still."
    },
    {
        "code": "ERROR 3",
        "keywords": ["clothing", "interfering", "sleeve", "error 3"],
        "solution": "Remove any clothing interfering with the arm cuff."
    }
]


# =========================================================
# MODEL INITIALIZATION
# =========================================================
# YOLO is used for detecting the BP display region.
# EasyOCR is used for extracting text from the cropped display.
with SuppressOutput():
    yolo_model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(LANGS, gpu=USE_GPU)


# =========================================================
# IMAGE PREPROCESSING FUNCTIONS
# =========================================================

def preprocess_otsu(img):
    """
    Applies Gaussian blur followed by Otsu thresholding.

    Input:
        img (numpy.ndarray): Cropped BGR image of BP display

    Returns:
        numpy.ndarray: Binary image optimized for OCR
    """
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.bitwise_not(thresh)


def preprocess_adaptive(img):
    """
    Applies CLAHE contrast enhancement and adaptive thresholding.

    Input:
        img (numpy.ndarray): Cropped BGR image

    Returns:
        numpy.ndarray: Binary image suitable for OCR
    """
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)
    return cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )


def preprocess_morph(img):
    """
    Applies Otsu thresholding followed by morphological dilation
    to thicken digit strokes.

    Input:
        img (numpy.ndarray): Cropped BGR image

    Returns:
        numpy.ndarray: Binary image
    """
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(cv2.bitwise_not(thresh), kernel, iterations=1)


def preprocess_tophat(img):
    """
    Uses a top-hat morphological transform to highlight
    bright digits on a dark background.

    Input:
        img (numpy.ndarray): Cropped BGR image

    Returns:
        numpy.ndarray: Binary image
    """
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.bitwise_not(thresh)


def preprocess_normalize_clahe(img):
    """
    Normalizes brightness, enhances contrast using CLAHE,
    and applies Otsu thresholding.

    Input:
        img (numpy.ndarray): Cropped BGR image

    Returns:
        numpy.ndarray: Binary image
    """
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(4.0, (8, 8))
    enhanced = clahe.apply(norm)
    _, thresh = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.bitwise_not(thresh)


# =========================================================
# OCR HELPER FUNCTIONS
# =========================================================

def extract_numbers(results):
    """
    Cleans OCR output and extracts valid numeric BP values.

    Input:
        results (list): EasyOCR output [(bbox, text, confidence), ...]

    Returns:
        list[int]: List of numeric values greater than 30
    """
    nums = []

    for _, text, _ in results:
        t = text.lower()

        # Remove known units and labels
        for unit in ALLOWED_UNITS:
            t = t.replace(unit, "")

        # Skip text that still contains letters
        if re.search(r'[a-z]', t):
            continue

        # Extract numeric substrings
        for n in re.findall(r'\d+', t):
            value = int(n)
            if value > 30:
                nums.append(value)

    return nums

def extract_numbers_with_xpos(results):
    """
    Extract numeric BP values along with their horizontal position (cx).

    Input:
        results (list): EasyOCR output [(bbox, text, confidence), ...]

    Returns:
        list[dict]: [{ "value": int, "cx": float }]
    """
    detections = []

    for bbox, text, _ in results:
        t = text.lower()

        # Remove known units and labels
        for unit in ALLOWED_UNITS:
            t = t.replace(unit, "")

        # Skip if letters still exist
        if re.search(r'[a-z]', t):
            continue

        nums = re.findall(r'\d+', t)
        if not nums:
            continue

        value = int(nums[0])
        if value <= 30:
            continue

        # Horizontal center (cx)
        xs = [p[0] for p in bbox]
        cx = sum(xs) / len(xs)

        detections.append({
            "value": value,
            "cx": cx
        })

    return detections


def identify_specific_error(full_text):
    """
    Matches OCR text against known BP monitor error patterns.

    Input:
        full_text (str): Combined OCR text

    Returns:
        tuple(str, str): (error_code, solution)
    """
    text_upper = full_text.upper()

    for error in KNOWN_ERRORS:
        for keyword in error["keywords"]:
            if keyword.upper() in text_upper:
                return error["code"], error["solution"]

    return "UNKNOWN ERROR"


# =========================================================
# CORE PIPELINE FUNCTIONS
# =========================================================

def get_best_crop(image_path):
    """
    Uses YOLO to detect the BP display region and returns
    the highest-confidence crop.

    Input:
        image_path (str): Path to input image file

    Returns:
        numpy.ndarray | None: Cropped BP display image or None
    """
    with SuppressOutput():
        image = cv2.imread(image_path)
        if image is None:
            return None

        results = yolo_model(image, conf=CONF_THRES, verbose=False)
        h, w, _ = image.shape

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            confidences = r.boxes.conf.cpu().numpy()
            best_idx = confidences.argmax()
            box = r.boxes[best_idx]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                return crop

    return None


def process_ocr(crop_img, image_name="Unknown"):
    """
    Runs OCR using multiple preprocessing strategies and
    returns structured JSON output.

    Input:
        crop_img (numpy.ndarray): Cropped BP display image
        image_name (str): Name of source image

    Returns:
        dict: JSON-compatible dictionary containing readings or error info
    """
    response = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_image": image_name,
        "status": "failed",
        "message": "",
        "SYS": None,
        "DIA": None,
        "HR": None,
        "error_code": None,
        "solution": None
    }

    if crop_img is None:
        response["message"] = "No cropped image provided"
        return response

    preprocessing_methods = [
        (lambda x: x, "Raw"),
        (preprocess_otsu, "Otsu"),
        (preprocess_adaptive, "Adaptive"),
        (preprocess_morph, "Morph"),
        (preprocess_tophat, "TopHat"),
        (preprocess_normalize_clahe, "Normalize")
    ]

    with SuppressOutput():
        for fn, name in preprocessing_methods:
            try:
                proc = crop_img if name == "Raw" else fn(crop_img)
                results = reader.readtext(proc, detail=1)
                full_text = " ".join([t for _, t, _ in results])

                # Detect BP monitor error screen
                if "ERROR" in full_text.upper():
                    code, solution = identify_specific_error(full_text)
                    response.update({
                        "status": "error",
                        "error_code": code,
                        "solution": solution,
                        "message": "BP monitor error detected"
                    })
                    return response

                detections = extract_numbers_with_xpos(results)

                # Successful BP reading requires exactly 3 spatially distinct values
                if len(detections) == 3:
                    # Sort left → right
                    detections = sorted(detections, key=lambda x: x["cx"])

                    sys_val = detections[0]["value"]   # LEFT
                    dia_val = detections[1]["value"]   # MIDDLE
                    hr_val  = detections[2]["value"]   # RIGHT

                    # Sanity check (basic physiology)
                    if sys_val > dia_val:
                        response.update({
                            "status": "success",
                            "SYS": sys_val,
                            "DIA": dia_val,
                            "HR": hr_val,
                            "message": "Readings extracted successfully"
                        })
                        return response


            except Exception:
                continue

    response["message"] = "Could not extract clear readings"
    return response