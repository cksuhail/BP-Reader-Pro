
# BP-Reader Pro

**BP-Reader Pro** is a Python-based computer vision system developed during an internship at **iHub Robotics, Edappally**.  
The system detects and extracts blood pressure readings from a **specific digital BP monitor** using a camera, YOLO-based object detection, and OCR.

It captures images, detects the BP display region, extracts readings (SYS, DIA, HR), identifies known device errors, and stores structured logs for review.


---

## Project Objective

- Automate BP monitor reading extraction using vision-based methods  
- Reduce manual transcription errors  
- Log readings and error states in structured JSON format  
- Serve as a proof-of-concept for medical device automation using computer vision  

---

## Technologies Used

- Python 3.12  
- Ultralytics YOLO – display detection  
- EasyOCR – digit recognition  
- OpenCV – camera access and preprocessing  
- NumPy  
- PyTorch  

---

## Hardware Assumptions

### Camera
- Any USB camera can be used  
- Proper focus is critical for accurate OCR  
- IMX335 was used during development (not mandatory)  

### BP Monitor
- Designed and tested for **one specific BP monitor model**  
- May require retraining for other models  

### Lighting
- Indoor stable lighting recommended  
- Avoid reflections and glare  
- Fixed camera position improves consistency  

---

## Platform Support

- Tested on: Windows  
- Expected to work on: Linux  

No OS-specific dependencies are used. Camera index adjustments may be required.

---

## Repository Structure
BP-Reader-Pro/
├── Code/
│ ├── backend.py
│ └── main.py
│
├── Models/
│ ├── Best_yolo_crop.pt
│ └── Last_yolo_crop.pt
│
├── Sample_Input/
│ └── captured_images/
│
├── Sample_Output/
│ ├── cropped_images/
│ └── logs/
│
├── requirements.txt
└── README.md
