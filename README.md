# BP-Reading

**BP-Reading** is a Python-based computer vision system developed during an internship at **iHub Robotics, Edappally**.
The project detects and extracts blood pressure readings from a **specific digital BP monitor** using a camera, YOLO-based object detection, and OCR.

The system captures images, detects the BP display region, extracts readings (SYS, DIA, HR), identifies known device errors, and stores structured logs for later review.

>  This is a technical automation and research prototype, **not a certified medical diagnostic system**.

---

## Project Objective

* Automate BP monitor reading extraction using vision-based methods
* Reduce manual transcription errors
* Log readings and error states in a structured, reviewable format
* Serve as a proof-of-concept for vision-based medical device automation

---

## Technologies Used

* **Python 3.12**
* **Ultralytics YOLO** – display detection
* **EasyOCR** – digit recognition
* **OpenCV** – camera access & image preprocessing
* **NumPy**
* **PyTorch**

---

## Hardware Assumptions

* **Camera**

  * Any USB camera can be used
  * Focus distance is critical — ensure the BP display is sharp and readable
  * IMX335 was used during development, but not mandatory

* **BP Monitor**

  * Designed and tested for **one specific BP monitor model**
  * Not guaranteed to generalize to other models without retraining

* **Lighting**

  * Indoor lighting recommended
  * Avoid reflections and glare on the BP display
  * Stable camera placement improves accuracy

---

## Platform Support

* **Tested on:** Windows
* **Expected to work on:** Linux

> No OS-specific code is used. Camera index and backend may need adjustment on Linux systems.

---

## Repository Structure

```
BP-READING/
├── Code/
│   ├── backend.py              # YOLO + OCR pipeline logic
│   └── main.py                 # Camera capture & user interaction loop
│
├── Models/
│   ├── Best_yolo_crop.pt       # Primary trained YOLO model
│   └── 2nd_Best_yolo_crop.pt   # Fallback YOLO model
│
├── Sample_input/
│   └── captured_images/        # Sample full-resolution input images
│
├── Sample_output/
│   ├── cropped_images/         # YOLO-cropped display regions
│   └── logs/                   # JSON logs (results & errors)
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmxdkm/BP-Reading.git
cd BP-READING
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> GPU acceleration depends on how **PyTorch** is installed.
> CPU-only execution works by default.

---

## How to Run

From the project root:

```bash
python Code/main.py
```

### Runtime flow:

1. Enter user details (name, age, sex)
2. Live camera preview starts
3. Controls:

   * **`c`** → Capture image and process
   * **`q`** → Quit application
4. Outputs are:

   * Printed in terminal
   * Saved as JSON logs
   * Stored along with original and cropped images

---

##  Camera Configuration

The application uses OpenCV to access a USB camera.
If multiple cameras are connected, or if you are using a camera with different capabilities, you may need to modify the camera configuration in `Code/main.py`.

```python
# USB camera index (change if multiple cameras are connected)
USB_CAM_INDEX = 0

# Camera resolution (set according to your camera capability)
CAM_WIDTH  = 2592
CAM_HEIGHT = 1944
```


## Output Details

For each capture, the system generates:

* Original captured image
* Cropped BP display image
* JSON log containing:

  * Timestamp
  * User metadata
  * SYS / DIA / HR values (if detected)
  * Error code and suggested solution (if detected)
  * Image file paths

Logs are designed for **later review, debugging, and validation**.

---

## Error Detection

The system attempts to detect known BP monitor error states such as:

* Improper cuff placement
* User movement during measurement
* Clothing interference

If detected, the output status is marked as `"error"` with a corresponding solution message.

---

## Data & Privacy

* All data is stored **locally**
* No cloud services or network communication
* User details (name, age, sex) are saved in log files
  → Handle responsibly if used beyond development or testing

---

## Disclaimer

This project is **not a medical device** and must not be used for clinical diagnosis or treatment decisions.
It is intended strictly for **engineering, automation, and research purposes**.

---

## Context

Developed during an internship at **iHub Robotics, Edappally** as part of a computer vision automation task.

