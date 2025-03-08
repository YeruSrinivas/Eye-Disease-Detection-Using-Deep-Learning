from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = load_model("evgg.h5")

# Define disease categories
index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Get uploaded file
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Process the image
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Make prediction
            preds = model.predict(x)
            pred_class = np.argmax(preds, axis=1)[0]
            result = index[pred_class]

            return render_template("result.html", filename=file.filename, result=result)

    return render_template("index.html")

@app.route("/display/<filename>")
def display_image(filename):
    return f'<img src="{os.path.join(app.config["UPLOAD_FOLDER"], filename)}" width="300">'

if __name__ == "__main__":
    app.run(debug=True)
