from flask import Flask, request, render_template_string, Response, redirect, url_for
import cv2
import numpy as np
import os
import sys
import uuid
import platform

# Ensure we can import modules from the local src directory when running app.py directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plate_detector import detect_plates
from detect_count import process_video, draw_label, VEHICLE_CLASSES
from tracker import CentroidTracker
from ultralytics import YOLO
from accident import AccidentDetector

app = Flask(__name__)

# Load YOLO model once for image annotations
YOLO_MODEL = YOLO(os.path.join(CURRENT_DIR, 'yolov8n.pt')) if os.path.exists(os.path.join(CURRENT_DIR, 'yolov8n.pt')) else YOLO('yolov8n.pt')

# Track uploaded video sources by id for streaming
VIDEO_SOURCES = {}
VIDEO_TARGET = {}
VIDEO_SNAPSHOT = {}
VIDEO_CONFIG = {}
ACCIDENT_SNAPSHOT = {}

HTML = '''
<!doctype html>
<title>Plate Detection</title>
<h2>Upload an image or video</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*,video/*">
  <label><input type=checkbox name=video value=1> Treat as video</label>
  <label style="margin-left: 10px;"><input type=checkbox name=ocr value=1> Enable OCR (ANPR)</label>
  <input type=submit value=Upload>
</form>
<p><a href="/live">Go to Live Detection</a></p>
{% if results %}
  <h3>Detected Plates:</h3>
  <ul>
  {% for plate in results %}
    <li>BBox: {{ plate['bbox'] }}, Text: {{ plate['text'] }}</li>
  {% endfor %}
  </ul>
{% endif %}
{% if image_url %}
  <h3>Processed Image:</h3>
  <img src="{{ image_url }}" style="width: 100%; max-width: 1200px; height: auto; border:2px solid #333;" />
{% endif %}
{% if video_url %}
  <h3>Processed Video:</h3>
  <video width="100%" style="max-width: 1200px; height: auto; border:2px solid #333;" controls>
    <source src="{{ video_url }}" type="video/mp4">
  </video>
  {% if counts %}
    <h3>Vehicle Counts:</h3>
    <p>IN: {{ counts['in'] }} | OUT: {{ counts['out'] }}</p>
  {% endif %}
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = None
    video_url = None
    image_url = None
    counts = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = file.filename.lower()
            is_video = ('video' in request.form) or any(filename.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv'])
            want_ocr = ('ocr' in request.form)

            if is_video:
                tmp_in = os.path.join(CURRENT_DIR, 'upload_video_tmp')
                os.makedirs(tmp_in, exist_ok=True)
                in_path = os.path.join(tmp_in, 'input_' + filename.replace(' ', '_'))
                file.save(in_path)

                vid = uuid.uuid4().hex
                VIDEO_SOURCES[vid] = in_path
                VIDEO_CONFIG[vid] = {'ocr': want_ocr}
                return redirect(url_for('watch_video', vid=vid))
            else:
                # Read uploaded bytes and decode as image
                file_bytes = file.read()
                img_bytes = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    results = [{'bbox': (0, 0, 0, 0), 'text': 'Failed to decode image. Please upload a valid image file.'}]
                else:
                    # 1) Vehicle detection using YOLO
                    try:
                        yolo_results = YOLO_MODEL.predict(img, conf=0.3, iou=0.5, verbose=False)
                        r = yolo_results[0]
                        boxes = r.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls not in VEHICLE_CLASSES:
                                continue
                            score = float(box.conf[0])
                            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                            label = f"{VEHICLE_CLASSES[cls]}: {int(score*100)}%"
                            draw_label(img, (x1b, y1b, x2b, y2b), label, color=(0, 255, 0), font_scale=0.6, thickness=2)
                    except Exception:
                        pass

                    # 2) Optional: license plate candidates (non-OCR) over whole image
                    plate_results = detect_plates(img, ocr=want_ocr)
                    if plate_results:
                        if results is None:
                            results = []
                        results.extend(plate_results)
                        for pr in plate_results:
                            x1, y1, x2, y2 = pr['bbox']
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            if pr.get('text'):
                                cv2.putText(img, pr['text'], (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    out_img_path = os.path.join(CURRENT_DIR, 'output.jpg')
                    ok = cv2.imwrite(out_img_path, img)
                    if ok:
                        image_url = '/image/output'
                    else:
                        results = [{'bbox': (0, 0, 0, 0), 'text': 'Failed to save processed image.'}]
    return render_template_string(HTML, results=results, video_url=video_url, image_url=image_url, counts=counts)


@app.route('/video/output')
def serve_output_video():
    # Serve the most recent processed video
    from flask import send_file
    out_path = os.path.join(CURRENT_DIR, 'output.mp4')
    if not os.path.exists(out_path):
        return 'No processed video found', 404
    return send_file(out_path, mimetype='video/mp4', as_attachment=False)


@app.route('/image/output')
def serve_output_image():
    from flask import send_file
    out_img_path = os.path.join(CURRENT_DIR, 'output.jpg')
    if not os.path.exists(out_img_path):
        return 'No processed image found', 404
    return send_file(out_img_path, mimetype='image/jpeg', as_attachment=False)


def gen_live(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')
    # Set camera resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    tracker = CentroidTracker(max_disappeared=60, max_distance=100)
    accident_detector = AccidentDetector(iou_threshold=0.15, dy_threshold=3.0, consecutive_frames=2)
    counts = {'in': 0, 'out': 0}
    object_tracks = {}
    counted_ids = set()
    line_initialized = False
    x1 = y1 = x2 = y2 = 0
    frame_skip = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                h, w = frame.shape[:2]
                if not line_initialized:
                    x1, y1, x2, y2 = 0, int(h * 0.6), w, int(h * 0.6)
                    line_initialized = True

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (640, 480))
                yolo_results = YOLO_MODEL.predict(small_frame, conf=0.2, iou=0.4, verbose=False, half=True, device='cpu')
                r = yolo_results[0]
                boxes = r.boxes
                rects = []
                classes = []
                scores = []
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls not in VEHICLE_CLASSES:
                        continue
                    score = float(box.conf[0])
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    rects.append((x1b, y1b, x2b, y2b))
                    classes.append(VEHICLE_CLASSES[cls])
                    scores.append(score)

                objects = tracker.update(rects)

                # draw counting line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                line_y = (y1 + y2) // 2

                # map object ids to nearest rect/class
                centroid_to_info = {}
                for oid, centroid in objects.items():
                    if len(rects) == 0:
                        continue
                    dists = [np.linalg.norm(np.array(centroid) - np.array(((rct[0]+rct[2])//2, (rct[1]+rct[3])//2))) for rct in rects]
                    idx = int(np.argmin(dists))
                    centroid_to_info[oid] = {
                        'centroid': centroid,
                        'bbox': rects[idx],
                        'class': classes[idx] if idx < len(classes) else 'vehicle',
                        'score': scores[idx] if idx < len(scores) else 0.0
                    }

                # build inputs for accident detector
                oid_to_dy = {}
                oid_to_bbox = {}
                oid_to_centroid_y = {}
                for oid, info in centroid_to_info.items():
                    c = info['centroid']
                    bbox = info['bbox']
                    cls_name = info['class']
                    cx, cy = int(c[0]), int(c[1])

                    # track history
                    if oid not in object_tracks:
                        object_tracks[oid] = []
                    object_tracks[oid].append(cy)
                    if len(object_tracks[oid]) > 10:
                        object_tracks[oid] = object_tracks[oid][-10:]
                    # dy magnitude as simple speed proxy
                    if len(object_tracks[oid]) >= 2:
                        oid_to_dy[oid] = abs(object_tracks[oid][-1] - object_tracks[oid][-2])
                    else:
                        oid_to_dy[oid] = 0
                    oid_to_bbox[oid] = bbox
                    oid_to_centroid_y[oid] = cy

                    conf_pct = int(info.get('score', 0) * 100)
                    label = f"{cls_name}: {conf_pct}%"
                    draw_label(frame, bbox, label, color=(0, 255, 0), font_scale=0.7, thickness=2)
                    cv2.putText(frame, f"ID {oid}", (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                    # crossing check
                    if oid not in counted_ids and len(object_tracks[oid]) >= 2:
                        prev_y = object_tracks[oid][-2]
                        cur_y = object_tracks[oid][-1]
                        if prev_y < line_y and cur_y >= line_y:
                            counts['in'] += 1
                            counted_ids.add(oid)
                        elif prev_y > line_y and cur_y <= line_y:
                            counts['out'] += 1
                            counted_ids.add(oid)

                # overlay totals
                cv2.putText(frame, f"IN: {counts['in']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"OUT: {counts['out']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # accident detection via module
                accident = accident_detector.update(oid_to_bbox, oid_to_centroid_y)
                if accident:
                    cv2.putText(frame, 'ACCIDENT DETECTED', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            except Exception:
                pass

            # Higher quality encoding for clearer video
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


@app.route('/live_feed')
def live_feed():
    return Response(gen_live(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live')
def live_page():
    live_html = '''
    <!doctype html>
    <title>Live Detection</title>
    <h2>Live Webcam Detection</h2>
    <p><a href="/">Back to Home</a></p>
    <img id="liveStream" src="/live_feed" style="width: 100%; max-width: 1600px; height: auto; border:2px solid #333;" />
    <br><br>
    <button onclick="toggleFullscreen()" style="padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">Toggle Fullscreen</button>
    <script>
    function toggleFullscreen() {
        const img = document.getElementById('liveStream');
        if (img.requestFullscreen) {
            img.requestFullscreen();
        } else if (img.webkitRequestFullscreen) {
            img.webkitRequestFullscreen();
        } else if (img.msRequestFullscreen) {
            img.msRequestFullscreen();
        }
    }
    </script>
    '''
    return render_template_string(live_html)


def gen_video_stream(video_path: str, vid: str | None = None, target_id: int | None = None):
    # Try different backends for better compatibility
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        print("Try converting to H.264 MP4 format")
        # Return a simple error image
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Video format not supported", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_img, "Try H.264 MP4", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    # Set video properties for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = CentroidTracker(max_disappeared=60, max_distance=100)
    accident_detector = AccidentDetector(iou_threshold=0.15, dy_threshold=3.0, consecutive_frames=2)
    counts = {'in': 0, 'out': 0}
    object_tracks = {}
    counted_ids = set()
    line_initialized = False
    x1 = y1 = x2 = y2 = 0
    frame_skip = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                h, w = frame.shape[:2]
                if not line_initialized:
                    x1, y1, x2, y2 = 0, int(h * 0.6), w, int(h * 0.6)
                    line_initialized = True

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (640, 480))
                yolo_results = YOLO_MODEL.predict(small_frame, conf=0.2, iou=0.4, verbose=False, half=True, device='cpu')
                r = yolo_results[0]
                boxes = r.boxes
                rects = []
                classes = []
                scores = []
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls not in VEHICLE_CLASSES:
                        continue
                    score = float(box.conf[0])
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    rects.append((x1b, y1b, x2b, y2b))
                    classes.append(VEHICLE_CLASSES[cls])
                    scores.append(score)

                objects = tracker.update(rects)

                # draw counting line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                line_y = (y1 + y2) // 2

                centroid_to_info = {}
                for oid, centroid in objects.items():
                    if len(rects) == 0:
                        continue
                    dists = [np.linalg.norm(np.array(centroid) - np.array(((rct[0]+rct[2])//2, (rct[1]+rct[3])//2))) for rct in rects]
                    idx = int(np.argmin(dists))
                    centroid_to_info[oid] = {
                        'centroid': centroid,
                        'bbox': rects[idx],
                        'class': classes[idx] if idx < len(classes) else 'vehicle',
                        'score': scores[idx] if idx < len(scores) else 0.0
                    }

                stop_stream = False
                oid_to_bbox = {}
                oid_to_centroid_y = {}
                for oid, info in centroid_to_info.items():
                    c = info['centroid']
                    bbox = info['bbox']
                    cls_name = info['class']
                    cx, cy = int(c[0]), int(c[1])

                    if oid not in object_tracks:
                        object_tracks[oid] = []
                    object_tracks[oid].append(cy)
                    if len(object_tracks[oid]) > 10:
                        object_tracks[oid] = object_tracks[oid][-10:]

                    conf_pct = int(info.get('score', 0) * 100)
                    label = f"{cls_name}: {conf_pct}%"
                    draw_label(frame, bbox, label, color=(0, 255, 0), font_scale=0.7, thickness=2)
                    cv2.putText(frame, f"ID {oid}", (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    oid_to_bbox[oid] = bbox
                    oid_to_centroid_y[oid] = cy

                    if oid not in counted_ids and len(object_tracks[oid]) >= 2:
                        prev_y = object_tracks[oid][-2]
                        cur_y = object_tracks[oid][-1]
                        if prev_y < line_y and cur_y >= line_y:
                            counts['in'] += 1
                            counted_ids.add(oid)
                        elif prev_y > line_y and cur_y <= line_y:
                            counts['out'] += 1
                            counted_ids.add(oid)

                    # If a target id is set and appears, snapshot and stop
                    if target_id is not None and oid == target_id and not stop_stream:
                        # save snapshot of current frame crop
                        try:
                            x1c, y1c, x2c, y2c = bbox
                            crop = frame[max(0,y1c):max(0,y2c), max(0,x1c):max(0,x2c)]
                            snap_dir = os.path.join(CURRENT_DIR, 'snapshots')
                            os.makedirs(snap_dir, exist_ok=True)
                            snap_path = os.path.join(snap_dir, f'{vid}_id{oid}.jpg')
                            if crop.size > 0:
                                cv2.imwrite(snap_path, crop)
                                if vid is not None:
                                    VIDEO_SNAPSHOT[vid] = snap_path
                        except Exception:
                            pass
                        stop_stream = True

                # overlay totals
                cv2.putText(frame, f"IN: {counts['in']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"OUT: {counts['out']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # ANPR overlay if enabled for this video
                cfg = VIDEO_CONFIG.get(vid, {}) if vid else {}
                want_ocr = bool(cfg.get('ocr', False))
                if want_ocr and rects:
                    # For each tracked object, run plate detection in its ROI and overlay text
                    for oid, info in centroid_to_info.items():
                        x1b, y1b, x2b, y2b = info['bbox']
                        pad_w = int((x2b - x1b) * 0.05)
                        pad_h = int((y2b - y1b) * 0.1)
                        rx1 = max(0, x1b - pad_w)
                        ry1 = max(0, y1b - pad_h)
                        rx2 = min(frame.shape[1] - 1, x2b + pad_w)
                        ry2 = min(frame.shape[0] - 1, y2b + pad_h)
                        roi = frame[ry1:ry2, rx1:rx2]
                        plate_results = detect_plates(roi, ocr=True)
                        for pr in plate_results:
                            pbx1, pby1, pbx2, pby2 = pr['bbox']
                            pbx1_f = rx1 + pbx1
                            pby1_f = ry1 + pby1
                            pbx2_f = rx1 + pbx2
                            pby2_f = ry1 + pby2
                            cv2.rectangle(frame, (pbx1_f, pby1_f), (pbx2_f, pby2_f), (0, 255, 255), 2)
                            if pr.get('text'):
                                cv2.putText(frame, pr['text'], (pbx1_f, max(0, pby1_f - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # accident detection via module
                accident = accident_detector.update(oid_to_bbox, oid_to_centroid_y)
                if accident:
                    cv2.putText(frame, 'ACCIDENT DETECTED', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            except Exception:
                pass

            # Higher quality encoding for clearer video
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            if 'stop_stream' in locals() and stop_stream:
                break
    finally:
        cap.release()


@app.route('/stream_feed/<vid>')
def stream_feed(vid):
    video_path = VIDEO_SOURCES.get(vid)
    if not video_path or not os.path.exists(video_path):
        return 'Video not found', 404
    target_param = request.args.get('target')
    try:
        target_id = int(target_param) if target_param is not None and target_param != '' else None
    except Exception:
        target_id = None
    VIDEO_TARGET[vid] = target_id
    return Response(gen_video_stream(video_path, vid, target_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/watch/<vid>', methods=['GET','POST'])
def watch_video(vid):
    if vid not in VIDEO_SOURCES:
        return 'Video not found', 404
    if request.method == 'POST':
        target = request.form.get('track_id', '').strip()
        # toggle OCR setting per video
        cfg = VIDEO_CONFIG.get(vid, {})
        cfg['ocr'] = ('ocr' in request.form)
        VIDEO_CONFIG[vid] = cfg
        return redirect(url_for('watch_video', vid=vid, target=target))
    target_q = request.args.get('target', '')
    cfg = VIDEO_CONFIG.get(vid, {})
    ocr_checked_attr = 'checked' if cfg.get('ocr') else ''
    page = '''
    <!doctype html>
    <title>Video Detection</title>
    <h2>Streaming processed video</h2>
    <p><a href="/">Back to Home</a></p>
    <form method="post" style="margin-bottom: 10px;">
      <label>Stop on ID: <input type="number" name="track_id" value="''' + (target_q or '') + '''" /></label>
      <label style="margin-left: 10px;"><input type="checkbox" name="ocr" ''' + ocr_checked_attr + ''' /> ANPR</label>
      <button type="submit">Set</button>
    </form>
    <img id="videoStream" src="/stream_feed/''' + vid + (('?target=' + target_q) if target_q else '') + '''" style="width: 100%; max-width: 1600px; height: auto; border:2px solid #333;" />
    <br><br>
    <button onclick="toggleFullscreen()" style="padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">Toggle Fullscreen</button>
    <script>
    function toggleFullscreen() {
        const img = document.getElementById('videoStream');
        if (img.requestFullscreen) {
            img.requestFullscreen();
        } else if (img.webkitRequestFullscreen) {
            img.webkitRequestFullscreen();
        } else if (img.msRequestFullscreen) {
            img.msRequestFullscreen();
        }
    }
    </script>
    <p>Snapshot (after stop): <a href="/snapshot/''' + vid + '''" target="_blank">open</a></p>
    '''
    return render_template_string(page)


@app.route('/snapshot/<vid>')
def snapshot(vid):
    from flask import send_file
    path = VIDEO_SNAPSHOT.get(vid)
    if not path or not os.path.exists(path):
        return 'Snapshot not available yet', 404
    return send_file(path, mimetype='image/jpeg', as_attachment=False)


@app.route('/test_video/<vid>')
def test_video(vid):
    """Test if video can be opened"""
    video_path = VIDEO_SOURCES.get(vid)
    if not video_path or not os.path.exists(video_path):
        return 'Video not found', 404
    
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return f"Video OK: {width}x{height}, {fps:.1f}fps, {frame_count} frames"
    else:
        return f"ERROR: Cannot open video {video_path}. Try converting to H.264 MP4 format."

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000, use_reloader=False)