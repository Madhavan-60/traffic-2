Traffic Monitoring System (ML)

This project provides a simple traffic monitoring pipeline using a pretrained YOLO model (Ultralytics) and OpenCV.

What you'll get
- Vehicle detection using YOLOv8 (car, truck, bus, motorcycle)
- Centroid-based tracking
- Line-crossing counting (entering/exiting)
- Overlayed video output and CSV logs

Quickstart
1. Create a virtual environment (recommended) and install dependencies:

   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2. Run detection on a video file or webcam:

   python src/detect_count.py --source path/to/video.mp4 --output output.mp4

Files
- `src/detect_count.py`: main detection and counting script
- `src/tracker.py`: simple centroid tracker and util functions
- `requirements.txt`: Python dependencies

OCR / Number-plate notes
- The project includes an optional license-plate detection + OCR feature. To enable OCR you need:
  - The Tesseract binary installed on your system (https://github.com/tesseract-ocr/tesseract). On Windows, install the Tesseract executable and add it to your PATH or set pytesseract.pytesseract.tesseract_cmd accordingly.
  - The Python package `pytesseract` (added to `requirements.txt`).

Usage example (detect plates without OCR):

   python src/detect_count.py --source path/to/video.mp4 --output output.mp4 --plates

To enable OCR (if pytesseract and Tesseract are available):

   python src/detect_count.py --source path/to/video.mp4 --output output.mp4 --plates --ocr

Notes & next steps
- To improve accuracy, fine-tune a YOLO model on your dataset
- Replace centroid tracker with DeepSORT for robust ID assignment
- Add speed estimation and traffic classification

Dependency notes
- If you see a pip deprecation or build warning for packages like `filterpy`, it's safe to skip them unless you need Kalman-based trackers. Prefer upgrading pip inside your virtualenv first (see Quickstart). If you do need `filterpy`, install it after upgrading pip or by using pip's PEP 517 build options:

   python -m pip install --upgrade pip setuptools wheel
   pip install --use-pep517 filterpy

Or add a `pyproject.toml` to the package source before building. For most users the scaffold works without `filterpy`.
