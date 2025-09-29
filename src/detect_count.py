import argparse
import time
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import CentroidTracker
from plate_detector import detect_plates, has_pytesseract

# classes of interest (COCO labels)
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


def draw_label(img, bbox, text, color=(0, 255, 0), font_scale=0.6, thickness=2):
    """Draw a rectangle with a filled label above it.
    bbox: (x1,y1,x2,y2)
    text: label text
    """
    x1, y1, x2, y2 = bbox
    # box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # label background
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 4
    lx1 = x1
    ly1 = max(0, y1 - h - pad * 2)
    lx2 = x1 + w + pad * 2
    ly2 = y1
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)

    # label text (white on colored background)
    text_x = x1 + pad
    text_y = y1 - pad
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Path to video file or webcam index')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save overlay video')
    parser.add_argument('--line', type=int, nargs=4, default=None, help='Counting line x1 y1 x2 y2')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Ultralytics model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--plates', action='store_true', help='Enable license plate detection for vehicle ROIs')
    parser.add_argument('--ocr', action='store_true', help='Run OCR on detected plates (requires pytesseract and tesseract binary)')
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source
    out_path = args.output

    # open source: try integer camera index first, otherwise treat as path or URL
    try:
        src = int(source)
        print(f'Opening camera index {src}')
    except Exception:
        src = source
        print(f'Opening video source: {src}')

    # if src is a path string, check it exists (helps catch bad paths)
    if isinstance(src, str):
        p = Path(src)
        if not p.exists():
            print(f'Error: video file not found: {src}')
            return
        # quick reject common image file types to avoid accidentally processing a single image
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        if p.suffix.lower() in image_exts:
            print(f'Error: the source appears to be an image file ({p.suffix}). Please provide a video file or a camera index.')
            return

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('Error opening video source', src)
        return
    # If the source is a file, check the reported frame count so we can detect single-image inputs
    if isinstance(src, str):
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            frame_count = 0
        if frame_count <= 1:
            # some video files may report 0 for streams; only warn/exit for file paths
            print(f'Error: the provided file appears to have {frame_count} frame(s). This may be an image or an invalid video. Provide a proper video file.')
            cap.release()
            return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # load model
    model = YOLO(args.model)

    tracker = CentroidTracker(max_disappeared=40, max_distance=60)
    counts = {'in': 0, 'out': 0}

    # counting line
    if args.line:
        x1, y1, x2, y2 = args.line
    else:
        x1, y1, x2, y2 = 0, int(height*0.6), width, int(height*0.6)

    csv_path = Path('counts.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp_s', 'frame', 'id', 'class', 'direction'])

    # per-object recent centroid history for simple direction detection
    object_tracks = {}  # id -> list of y positions (most recent last)
    counted_ids = set()

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
            r = results[0]
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

            # build mapping from object id to nearest bbox/class
            centroid_to_info = {}
            for oid, centroid in objects.items():
                if len(rects) == 0:
                    continue
                dists = [np.linalg.norm(np.array(centroid) - np.array(((r[0]+r[2])//2, (r[1]+r[3])//2))) for r in rects]
                idx = int(np.argmin(dists))
                centroid_to_info[oid] = {
                    'centroid': centroid,
                    'bbox': rects[idx],
                    'class': classes[idx],
                    'score': scores[idx]
                }

            # draw counting line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_y = (y1 + y2) // 2

            # process tracked objects for crossing
            for oid, info in centroid_to_info.items():
                c = info['centroid']
                bbox = info['bbox']
                cls_name = info['class']
                cx, cy = int(c[0]), int(c[1])

                # update track history
                if oid not in object_tracks:
                    object_tracks[oid] = []
                object_tracks[oid].append(cy)
                # keep only last 10 positions
                if len(object_tracks[oid]) > 10:
                    object_tracks[oid] = object_tracks[oid][-10:]

                # draw bbox and label with confidence and id
                conf_pct = int(info.get('score', 0) * 100)
                label = f"{cls_name}: {conf_pct}%"
                draw_label(frame, bbox, label, color=(0,255,0), font_scale=0.5, thickness=1)
                cv2.putText(frame, f"ID {oid}", (bbox[0], bbox[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
                cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

                # check crossing event
                if oid not in counted_ids and len(object_tracks[oid]) >= 2:
                    prev_y = object_tracks[oid][-2]
                    cur_y = object_tracks[oid][-1]
                    # crossing downward (prev above line, current below or on)
                    if prev_y < line_y and cur_y >= line_y:
                        counts['in'] += 1
                        counted_ids.add(oid)
                        ts = frame_idx / fps
                        csv_writer.writerow([f"{ts:.2f}", frame_idx, oid, cls_name, 'in'])
                    # crossing upward
                    elif prev_y > line_y and cur_y <= line_y:
                        counts['out'] += 1
                        counted_ids.add(oid)
                        ts = frame_idx / fps
                        csv_writer.writerow([f"{ts:.2f}", frame_idx, oid, cls_name, 'out'])

                # optional: detect license plates inside vehicle bbox
                if args.plates:
                    # expand bbox slightly to include plate area
                    x1b, y1b, x2b, y2b = bbox
                    pad_w = int((x2b - x1b) * 0.05)
                    pad_h = int((y2b - y1b) * 0.1)
                    rx1 = max(0, x1b - pad_w)
                    ry1 = max(0, y1b - pad_h)
                    rx2 = min(width - 1, x2b + pad_w)
                    ry2 = min(height - 1, y2b + pad_h)
                    roi = frame[ry1:ry2, rx1:rx2]
                    plate_results = detect_plates(roi, ocr=(args.ocr and has_pytesseract()))
                    for pr in plate_results:
                        pbx1, pby1, pbx2, pby2 = pr['bbox']
                        # bbox coordinates are relative to roi; convert to frame coords
                        pbx1_f = rx1 + pbx1
                        pby1_f = ry1 + pby1
                        pbx2_f = rx1 + pbx2
                        pby2_f = ry1 + pby2
                        # draw plate box
                        cv2.rectangle(frame, (pbx1_f, pby1_f), (pbx2_f, pby2_f), (0, 255, 255), 2)
                        if pr.get('text'):
                            cv2.putText(frame, pr['text'], (pbx1_f, pby1_f - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

            # overlay counts
            cv2.putText(frame, f"IN: {counts['in']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
            cv2.putText(frame, f"OUT: {counts['out']}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)

            # write frame
            writer.write(frame)

            # optional: show in fullscreen
            cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Finished')
    finally:
        cap.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


def process_video(
    source_path: str,
    output_path: str = 'output.mp4',
    counting_line: tuple | None = None,
    model_path: str = 'yolov8n.pt',
    conf: float = 0.3,
    iou: float = 0.5,
    detect_plates_in_rois: bool = False,
    run_ocr: bool = False,
):
    """Process a video file, overlay detections and counts, and write output video.

    Returns a dict with keys: {'output_path': str, 'counts': {'in': int, 'out': int}}
    """
    p = Path(source_path)
    if not p.exists():
        return {'error': f'Video file not found: {source_path}'}

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {'error': f'Error opening video source {source_path}'}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        model = YOLO(model_path)
        tracker = CentroidTracker(max_disappeared=40, max_distance=60)
        counts = {'in': 0, 'out': 0}

        if counting_line:
            x1, y1, x2, y2 = counting_line
        else:
            x1, y1, x2, y2 = 0, int(height * 0.6), width, int(height * 0.6)

        object_tracks = {}
        counted_ids = set()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            r = results[0]
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

            centroid_to_info = {}
            for oid, centroid in objects.items():
                if len(rects) == 0:
                    continue
                dists = [np.linalg.norm(np.array(centroid) - np.array(((r[0] + r[2]) // 2, (r[1] + r[3]) // 2))) for r in rects]
                idx = int(np.argmin(dists))
                centroid_to_info[oid] = {
                    'centroid': centroid,
                    'bbox': rects[idx],
                    'class': classes[idx],
                    'score': scores[idx]
                }

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_y = (y1 + y2) // 2

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
                draw_label(frame, bbox, label, color=(0, 255, 0), font_scale=0.5, thickness=1)
                cv2.putText(frame, f"ID {oid}", (bbox[0], bbox[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                if oid not in counted_ids and len(object_tracks[oid]) >= 2:
                    prev_y = object_tracks[oid][-2]
                    cur_y = object_tracks[oid][-1]
                    if prev_y < line_y and cur_y >= line_y:
                        counts['in'] += 1
                        counted_ids.add(oid)
                    elif prev_y > line_y and cur_y <= line_y:
                        counts['out'] += 1
                        counted_ids.add(oid)

                if detect_plates_in_rois:
                    x1b, y1b, x2b, y2b = bbox
                    pad_w = int((x2b - x1b) * 0.05)
                    pad_h = int((y2b - y1b) * 0.1)
                    rx1 = max(0, x1b - pad_w)
                    ry1 = max(0, y1b - pad_h)
                    rx2 = min(width - 1, x2b + pad_w)
                    ry2 = min(height - 1, y2b + pad_h)
                    roi = frame[ry1:ry2, rx1:rx2]
                    plate_results = detect_plates(roi, ocr=(run_ocr and has_pytesseract()))
                    for pr in plate_results:
                        pbx1, pby1, pbx2, pby2 = pr['bbox']
                        pbx1_f = rx1 + pbx1
                        pby1_f = ry1 + pby1
                        pbx2_f = rx1 + pbx2
                        pby2_f = ry1 + pby2
                        cv2.rectangle(frame, (pbx1_f, pby1_f), (pbx2_f, pby2_f), (0, 255, 255), 2)
                        if pr.get('text'):
                            cv2.putText(frame, pr['text'], (pbx1_f, pby1_f - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"IN: {counts['in']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {counts['out']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            writer.write(frame)

        return {'output_path': output_path, 'counts': counts}
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            writer.release()
        except Exception:
            pass
