import streamlit as st
import pickle
import pandas as pd
import numpy as np

# โหลดโมเดล
try:
    # พยายามโหลดโมเดลเดิม
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.warning("กำลังสร้างโมเดลใหม่...")
    
    # ถ้าโหลดไม่สำเร็จ สร้างโมเดลใหม่
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    
    # โหลดข้อมูล
    try:
        df = pd.read_csv("Diabetes-dataset.csv")
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        
        # สร้างโมเดลใหม่
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        st.success("สร้างโมเดลใหม่เรียบร้อยแล้ว")
    except Exception as e:
        st.error(f"ไม่สามารถสร้างโมเดลใหม่ได้: {e}")

# สร้างหน้าเว็บ
st.title("ระบบประเมินความเสี่ยงเบาหวาน")

# สร้างฟอร์มรับข้อมูล
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("จำนวนการตั้งครรภ์", min_value=0, max_value=20)
    glucose = st.number_input("ระดับน้ำตาลในเลือด (Glucose)", min_value=0, max_value=300)
    blood_pressure = st.number_input("ความดันโลหิต (Blood Pressure)", min_value=0, max_value=200)
    skin_thickness = st.number_input("ความหนาผิวหนัง (Skin Thickness)", min_value=0, max_value=100)

with col2:
    insulin = st.number_input("ระดับอินซูลิน (Insulin)", min_value=0, max_value=1000)
    bmi = st.number_input("ดัชนีมวลกาย (BMI)", min_value=0.0, max_value=100.0, step=0.1)
    dpf = st.number_input("ค่าความเสี่ยงทางพันธุกรรม", min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input("อายุ", min_value=0, max_value=120)

# ปุ่มทำนาย
if st.button("ประเมินความเสี่ยง", type="primary"):
    # เตรียมข้อมูลสำหรับทำนาย
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # ทำนาย
    prediction = model.predict(features)[0]
    
    # แสดงผลลัพธ์
    if prediction == 1:
        st.error("⚠️ คุณอาจมีความเสี่ยงที่จะเป็น **โรคเบาหวาน**")
    else:
        st.success("✅ คุณไม่มีแนวโน้มที่จะเป็น **โรคเบาหวาน**")