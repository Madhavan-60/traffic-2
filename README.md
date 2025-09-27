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



Dependency notes
- If you see a pip deprecation or build warning for packages like `filterpy`, it's safe to skip them unless you need Kalman-based trackers. Prefer upgrading pip inside your virtualenv first (see Quickstart). If you do need `filterpy`, install it after upgrading pip or by using pip's PEP 517 build options:

   python -m pip install --upgrade pip setuptools wheel
   pip install --use-pep517 filterpy

Or add a `pyproject.toml` to the package source before building. For most users the scaffold works without `filterpy`.
