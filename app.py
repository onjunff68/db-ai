from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# โหลดโมเดล
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # รับค่าฟีเจอร์จากฟอร์ม
            features = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"]),
            ]
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"เกิดข้อผิดพลาด: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
