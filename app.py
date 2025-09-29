from flask import Flask, request, render_template_string
import cv2
import numpy as np
from plate_detector import detect_plates

app = Flask(__name__)

HTML = '''
<!doctype html>
<title>Plate Detection</title>
<h2>Upload an image for plate detection</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if results %}
  <h3>Detected Plates:</h3>
  <ul>
  {% for plate in results %}
    <li>BBox: {{ plate['bbox'] }}, Text: {{ plate['text'] }}</li>
  {% endfor %}
  </ul>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read image from upload
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            results = detect_plates(img, ocr=False)
    return render_template_string(HTML, results=results)

if __name__ == '__main__':
    app.run(debug=True)