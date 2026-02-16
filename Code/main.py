"""
This script is the main entry point for the BP reading system.

It captures images from a USB camera, saves the raw image, uses the backend
pipeline to detect the BP monitor display, extracts SYS/DIA/HR readings via OCR,
and stores the final results as structured JSON logs with user metadata.
"""

# =========================================================
# IMPORTS
# =========================================================
import backend
import os
import json
import cv2
from datetime import datetime


# =========================================================
# USER CONFIGURATION
# =========================================================

# Base directory where all outputs will be stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..")

# Sub-directories for different outputs
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, "Sample_Input", "captured_images")
CROPPED_DIR    = os.path.join(SAVE_DIR, "Sample_Output", "cropped_images")
LOG_FOLDER     = os.path.join(SAVE_DIR, "Sample_Output", "logs")

# USB camera index (change if multiple cameras are connected)
USB_CAM_INDEX = 0

# IMX335 maximum supported resolution
CAM_WIDTH  = 2592
CAM_HEIGHT = 1944


# =========================================================
# DIRECTORY SETUP
# =========================================================
# Ensure required folders exist before runtime
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)


# =========================================================
# USER INPUT FUNCTIONS
# =========================================================

def get_user_info():
    """
    Collects basic user information from the terminal.

    Input:
        None (interactive terminal input)

    Returns:
        tuple(str, str, str): (name, age, sex)
    """
    name = input("Enter name: ").strip().lower().replace(" ", "_")
    age  = input("Enter age: ").strip()
    sex  = input("Enter sex (M/F/O): ").strip().upper()
    return name, age, sex


# =========================================================
# LOGGING FUNCTION
# =========================================================

def save_log(data, user_name, user_age, ts):
    """
    Saves the OCR result and metadata into a JSON file.

    Input:
        data (dict): Final OCR and metadata dictionary
        user_name (str): User name
        user_age (str): User age
        ts (str): Timestamp string

    Returns:
        None (writes JSON file to disk)
    """
    filename = f"{user_name}_{user_age}_{ts}.json"
    file_path = os.path.join(LOG_FOLDER, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([data], f, indent=4)


# =========================================================
# IMAGE PROCESSING PIPELINE
# =========================================================

def process_image(image_path, user_name, user_age, user_sex, ts):
    """
    Runs the full backend pipeline on a captured image.

    Steps:
        1. Detect BP display using YOLO
        2. Save cropped display image
        3. Run OCR to extract readings
        4. Add user and file metadata
        5. Save final JSON log
        6. Print result to console

    Input:
        image_path (str): Path to captured image
        user_name (str): User name
        user_age (str): User age
        user_sex (str): User sex
        ts (str): Timestamp string

    Returns:
        None
    """
    img_name = os.path.basename(image_path)

    # Step 1: Detect and crop BP display using YOLO
    cropped_image = backend.get_best_crop(image_path)
    if cropped_image is None:
        print("No BP device detected in image")
        return

    # Step 2: Save cropped image
    crop_name = f"{user_name}_{user_age}_crop_{ts}.png"
    crop_path = os.path.join(CROPPED_DIR, crop_name)
    cv2.imwrite(crop_path, cropped_image)

    # Step 3: Run OCR on cropped image
    full_json_log = backend.process_ocr(
        cropped_image,
        image_name=img_name
    )

    # Step 4: Add user details and file metadata
    full_json_log.update({
        "name": user_name,
        "age": user_age,
        "sex": user_sex,
        "timestamp": ts,
        "source_image": image_path,
        "cropped_image": crop_path
    })

    # Step 5: Save JSON log to disk
    save_log(full_json_log, user_name, user_age, ts)

    # Step 6: Display result in terminal
    if full_json_log.get("status") == "success":
        print(
            f"SYS:{full_json_log.get('SYS')} "
            f"DIA:{full_json_log.get('DIA')} "
            f"HR:{full_json_log.get('HR')}"
        )
    else:
        print("Detection Failed")
        print(full_json_log.get("solution", ""))


# =========================================================
# MAIN APPLICATION LOOP
# =========================================================

def main():
    """
    Initializes the camera, handles live preview,
    captures images on key press, and triggers processing.

    Input:
        None

    Returns:
        None
    """
    # Collect user information once at startup
    user_name, user_age, user_sex = get_user_info()

    # Open USB camera
    cap = cv2.VideoCapture(USB_CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("USB Camera not accessible")

    # Configure camera resolution and format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Read back actual resolution set by camera
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Resolution: {actual_w} x {actual_h}")

    print("Press 'c' to CAPTURE")
    print("Press 'q' to QUIT")

    # Live camera loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for preview window
        preview = cv2.resize(frame, (960, 720))
        cv2.imshow("IMX335 Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        # Capture image
        if key == ord('c'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            image_name = f"{user_name}_{user_age}_{ts}.jpg"
            image_path = os.path.join(IMAGE_SAVE_DIR, image_name)

            # Save original full-resolution image
            cv2.imwrite(
                image_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            print(f"Captured: {image_name}")

            # Process captured image
            process_image(
                image_path,
                user_name,
                user_age,
                user_sex,
                ts
            )

        # Quit application
        elif key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()