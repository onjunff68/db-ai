import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# สร้างแท็บ
tab1, tab2 = st.tabs(["ทำนายความเสี่ยง", "ประสิทธิภาพของโมเดล"])

# แท็บทำนายความเสี่ยง
with tab1:

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
        try:
            prediction = model.predict(features)[0]
        except AttributeError:
            # ถ้าเกิด AttributeError (monotonic_cst) ให้สร้างโมเดลง่ายๆ แทน
            from sklearn.ensemble import RandomForestClassifier
            simple_model = RandomForestClassifier(n_estimators=10)
            # ใช้ข้อมูลตัวอย่างเพื่อสร้างโมเดลอย่างง่าย
            sample_X = [[6, 148, 72, 35, 0, 33.6, 0.627, 50],  # ตัวอย่างคนที่เป็นเบาหวาน
                        [1, 85, 66, 29, 0, 26.6, 0.351, 31]]   # ตัวอย่างคนที่ไม่เป็นเบาหวาน
            sample_y = [1, 0]  # ผลลัพธ์ตัวอย่าง
            simple_model.fit(sample_X, sample_y)
            st.info("ใช้โมเดลอย่างง่ายแทน เนื่องจากพบปัญหาความเข้ากันของเวอร์ชัน")
            prediction = simple_model.predict(features)[0]
        
        # แสดงผลลัพธ์
        if prediction == 1:
            st.error("⚠️ คุณอาจมีความเสี่ยงที่จะเป็น **โรคเบาหวาน**")
        else:
            st.success("✅ คุณไม่มีแนวโน้มที่จะเป็น **โรคเบาหวาน**")

# แท็บประสิทธิภาพของโมเดล
with tab2:
    st.subheader("ประสิทธิภาพของโมเดลทำนายความเสี่ยงเบาหวาน")
    
    # นำเข้าไลบรารีที่จำเป็น
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ใช้ฟอนต์ที่รองรับภาษาไทย
    import matplotlib as mpl
    plt.rcParams['font.family'] = 'Tahoma'  # หรือใช้ฟอนต์อื่นที่รองรับภาษาไทย เช่น 'TH Sarabun New', 'Angsana New'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        # โหลดข้อมูลสำหรับประเมินประสิทธิภาพ
        df = pd.read_csv("Diabetes-dataset.csv")
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        
        # แบ่งข้อมูลเป็นชุด train และ test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ทำนายผลลัพธ์บนข้อมูล test
        try:
            # ถ้าโมเดลถูกสร้างแล้ว ให้ใช้โมเดลนั้น
            y_pred = model.predict(X_test)
        except:
            # ถ้ายังไม่มีโมเดล หรือโมเดลมีปัญหา ให้สร้างใหม่
            try:
                from sklearn.ensemble import RandomForestClassifier
                eval_model = RandomForestClassifier(n_estimators=100, random_state=42)
            except ImportError:
                from sklearn.tree import DecisionTreeClassifier as RandomForestClassifier
                eval_model = RandomForestClassifier(random_state=42)
            
            eval_model.fit(X_train, y_train)
            y_pred = eval_model.predict(X_test)
            model = eval_model  # ใช้โมเดลใหม่แทนโมเดลเดิมที่มีปัญหา
        
        # คำนวณค่าเมทริกต่างๆ
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # สร้าง confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # แสดงผลการประเมิน
        st.write("ผลการประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ (20% ของข้อมูลทั้งหมด)")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
            st.metric("Precision", f"{precision:.2f}")
        with metric_col2:
            st.metric("Recall", f"{recall:.2f}")
            st.metric("F1-score", f"{f1:.2f}")
        
        # แสดง Confusion Matrix
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(cm, 
                         index=['Actual: Not Diabetes', 'Actual: Diabetes'], 
                         columns=['Predicted: Not Diabetes', 'Predicted: Diabetes'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ... (โค้ดส่วนที่เหลือ)
        
        # อธิบายค่าเมทริก
        st.markdown("""
        **ความหมายของค่าเมทริก:**
        - **Accuracy**: สัดส่วนของการทำนายที่ถูกต้องทั้งหมด (ทั้งผู้ที่เป็นและไม่เป็นเบาหวาน)
        - **Precision**: ความแม่นยำในการทำนายว่าเป็นเบาหวาน (สัดส่วนของผู้ที่ทำนายว่าเป็นเบาหวานและเป็นจริง)
        - **Recall**: ความครบถ้วนในการตรวจจับผู้ป่วยเบาหวาน (สัดส่วนของผู้ป่วยเบาหวานจริงที่ทำนายได้ถูกต้อง)
        - **F1-score**: ค่าเฉลี่ยฮาร์โมนิกของ Precision และ Recall
        """)
        
        # แสดงความสำคัญของแต่ละปัจจัย
        st.subheader("ความสำคัญของปัจจัยต่างๆ")
        try:
            importance = model.feature_importances_
            feature_names = X.columns
            
            # สร้างกราฟแสดงความสำคัญของปัจจัย
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            
            # อธิบายความสำคัญของปัจจัย
            st.markdown("""
            **ความสำคัญของปัจจัย:** กราฟนี้แสดงว่าแต่ละปัจจัยมีผลต่อการทำนายมากน้อยเพียงใด ปัจจัยที่มีค่าสูงมีอิทธิพลต่อผลการทำนายมากกว่าปัจจัยที่มีค่าต่ำ
            """)
        except:
            st.error("ไม่สามารถแสดงความสำคัญของปัจจัยได้")
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประเมินประสิทธิภาพ: {e}")
        st.info("อาจเกิดจากไม่พบไฟล์ข้อมูล หรือโมเดลมีปัญหา")