import cv2
import numpy as np
try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    pytesseract = None
    _HAS_PYTESSERACT = False


def _normalize_box(x, y, w, h, img_w, img_h):
    # clamp to image
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w - 1, x + w)
    y2 = min(img_h - 1, y + h)
    return (int(x1), int(y1), int(x2), int(y2))


def detect_plates(img, ocr=False, debug=False):
    """Detect candidate license plates in an image.

    Args:
        img: BGR image (numpy array)
        ocr: if True and pytesseract is available, return OCR text for each plate
        debug: if True, return intermediate debug images (not used currently)

    Returns: list of dicts: [{'bbox': (x1,y1,x2,y2), 'text': 'ABC123' or ''}, ...]
    """
    results = []
    if img is None:
        return results

    h_img, w_img = img.shape[:2]

    # convert to grayscale and apply bilateral filter to preserve edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # edge detection
    edged = cv2.Canny(gray, 30, 200)

    # find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # license plates are often rectangles (4 points)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # aspect ratio and size filtering - typical plate aspect ratio ~2-5
            if w == 0 or h == 0:
                continue
            ar = w / float(h)
            if ar < 1.5 or ar > 6.0:
                continue

            # ignore tiny boxes
            if w < 60 or h < 15:
                continue

            # ensure box is within image
            bx1, by1, bx2, by2 = _normalize_box(x - 5, y - 5, w + 10, h + 10, w_img, h_img)
            plate_roi = img[by1:by2, bx1:bx2]

            text = ''
            if ocr and _HAS_PYTESSERACT:
                # basic preprocessing for OCR
                plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.resize(plate_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                plate_gray = cv2.bilateralFilter(plate_gray, 9, 75, 75)
                _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                try:
                    text = pytesseract.image_to_string(plate_thresh, config=custom_config)
                    text = ''.join(ch for ch in text if ch.isalnum())
                except Exception:
                    text = ''

            results.append({'bbox': (bx1, by1, bx2, by2), 'text': text})

    return results


def has_pytesseract():
    return _HAS_PYTESSERACT

# Example usage:
img = cv2.imread('car.jpg')  # Replace with your image filename
results = detect_plates(img, ocr=False)  # Set ocr=True if you want OCR

for plate in results:
    print("Plate bbox:", plate['bbox'], "Text:", plate['text'])
    # Draw rectangle for visualization
    x1, y1, x2, y2 = plate['bbox']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
    cv2.putText(img, plate['text'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

cv2.imshow('Detected Plates', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update path as needed
print(pytesseract.get_tesseract_version())
