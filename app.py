from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model_path = "/content/drive/MyDrive/miniproject/your_model1.h5"
model = load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        temp_path = 'temp_image.png'
        uploaded_file.save(temp_path)
        
        img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = cv2.bitwise_not(img)
        img = cv2.dilate(img, None, iterations=1)
        
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        rects = sorted(rects, key=lambda x: sum(x[:2]))
        
        predicted_digits = ""
        for i, rect in enumerate(rects):
            x, y, w, h = rect
            digit = img[y:y+h, x:x+w]
            digit = cv2.copyMakeBorder(digit, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            digit = cv2.resize(digit, (28, 28))
            digit = digit.astype("float32") / 255.0
            digit = np.expand_dims(digit, axis=-1)
            digit = np.expand_dims(digit, axis=0)
            prediction = model.predict(digit)
            digit_class = np.argmax(prediction)
            predicted_digits += str(digit_class)
        
        return render_template('index.html', predicted_digits=predicted_digits)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
