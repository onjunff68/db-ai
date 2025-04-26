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
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # ใช้ฟอนต์ที่รองรับภาษาไทย
    import matplotlib as mpl
    plt.rcParams['font.family'] = 'Tahoma'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        # โหลดข้อมูลสำหรับประเมินประสิทธิภาพ
        df = pd.read_csv("Diabetes-dataset.csv")
        
        # ตรวจสอบและแสดงข้อมูลเบื้องต้น
        st.write(f"ข้อมูลทั้งหมด: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
        
        # แสดงตัวอย่างข้อมูล
        st.write("ตัวอย่างข้อมูล:")
        st.dataframe(df.head(5))
        
        # แสดงสัดส่วนของคลาส
        class_counts = df["Outcome"].value_counts()
        st.write("สัดส่วนของคลาส:")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='Outcome', data=df, ax=ax)
        ax.set_xlabel('ผลลัพธ์ (0 = ไม่เป็นเบาหวาน, 1 = เป็นเบาหวาน)')
        ax.set_ylabel('จำนวน')
        st.pyplot(fig)
        
        # แยกตัวแปรต้นและตัวแปรตาม
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        
        # เตรียมข้อมูลเพิ่มเติม (จัดการกับค่าสูญหาย หากมี)
        for col in X.columns:
            # แทนที่ค่า 0 ในคอลัมน์ที่ไม่ควรเป็น 0 ด้วยค่าเฉลี่ย
            if col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
                X.loc[X[col] == 0, col] = np.nan
                X[col].fillna(X[col].mean(), inplace=True)
        
        # สเกลข้อมูล
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # แบ่งข้อมูลเป็น 3 ส่วน: train, validation, test (60-20-20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # ประเมินด้วย cross-validation
        st.subheader("ประเมินโมเดลด้วย Cross-Validation")
        
        # สร้างโมเดล
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,  # จำกัดความลึกเพื่อลด overfitting
            min_samples_split=5,  # ปรับพารามิเตอร์เพื่อลด overfitting
            random_state=42
        )
        
        # ทำ cross-validation เพื่อประเมินความแม่นยำ
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        st.write(f"ความแม่นยำเฉลี่ยจาก 5-fold Cross-Validation: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # แสดงคะแนนแต่ละ fold
        fold_scores = pd.DataFrame({
            'Fold': [f"Fold {i+1}" for i in range(len(cv_scores))],
            'Accuracy': cv_scores
        })
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x='Fold', y='Accuracy', data=fold_scores, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title('ความแม่นยำในแต่ละ Fold')
        st.pyplot(fig)
        
        # เทรนโมเดลกับชุด train และทดสอบกับชุด validation
        model.fit(X_train, y_train)
        
        # ทำนายผลลัพธ์บนชุด validation
        y_val_pred = model.predict(X_val)
        
        # คำนวณค่าเมทริกต่างๆ บนชุด validation
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # แสดงผลการประเมินบนชุด validation
        st.subheader("ผลการประเมินประสิทธิภาพบนชุดข้อมูล Validation")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{val_accuracy:.2f}")
            st.metric("Precision", f"{val_precision:.2f}")
        with metric_col2:
            st.metric("Recall", f"{val_recall:.2f}")
            st.metric("F1-score", f"{val_f1:.2f}")
        
        # แสดง Confusion Matrix สำหรับชุด validation
        st.subheader("Confusion Matrix (Validation Set)")
        cm_val = confusion_matrix(y_val, y_val_pred)
        cm_val_df = pd.DataFrame(cm_val, 
                             index=['Actual: Not Diabetes', 'Actual: Diabetes'], 
                             columns=['Predicted: Not Diabetes', 'Predicted: Diabetes'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_val_df, annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ทำนายผลลัพธ์บนชุด test
        y_test_pred = model.predict(X_test)
        
        # คำนวณค่าเมทริกต่างๆ บนชุด test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # แสดงผลการประเมินบนชุด test
        st.subheader("ผลการประเมินประสิทธิภาพบนชุดข้อมูล Test (ข้อมูลที่โมเดลไม่เคยเห็นมาก่อน)")
        
        test_col1, test_col2 = st.columns(2)
        with test_col1:
            st.metric("Accuracy", f"{test_accuracy:.2f}")
            st.metric("Precision", f"{test_precision:.2f}")
        with test_col2:
            st.metric("Recall", f"{test_recall:.2f}")
            st.metric("F1-score", f"{test_f1:.2f}")
        
        # แสดง Confusion Matrix สำหรับชุด test
        st.subheader("Confusion Matrix (Test Set)")
        cm_test = confusion_matrix(y_test, y_test_pred)
        cm_test_df = pd.DataFrame(cm_test, 
                             index=['Actual: Not Diabetes', 'Actual: Diabetes'], 
                             columns=['Predicted: Not Diabetes', 'Predicted: Diabetes'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_test_df, annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
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
        except Exception as e:
            st.error(f"ไม่สามารถแสดงความสำคัญของปัจจัยได้: {e}")
        
        # อธิบายเพิ่มเติมเกี่ยวกับความน่าเชื่อถือของผลลัพธ์
        st.subheader("หมายเหตุเกี่ยวกับการประเมินประสิทธิภาพ")
        st.info("""
        การประเมินประสิทธิภาพของโมเดลอย่างน่าเชื่อถือควรทำโดย:
        1. **ข้อมูลที่สมดุล**: ควรมีจำนวนตัวอย่างของแต่ละคลาส (เป็นหรือไม่เป็นเบาหวาน) ในสัดส่วนที่เหมาะสม
        2. **Cross-validation**: ช่วยให้มั่นใจว่าผลการประเมินไม่ได้เกิดจากการสุ่มแบ่งข้อมูลที่เอื้อประโยชน์เป็นพิเศษ
        3. **การแยกชุดข้อมูล**: การแบ่งข้อมูลเป็น train/validation/test ช่วยให้เราสามารถปรับแต่งโมเดลและประเมินด้วยข้อมูลที่ไม่เคยเห็นมาก่อนจริงๆ
        4. **การจัดการค่าสูญหาย**: เราได้แทนที่ค่า 0 ที่ไม่สมเหตุสมผลในตัวแปรบางตัว เช่น น้ำตาลในเลือด (Glucose) เพื่อให้โมเดลเรียนรู้ได้ดีขึ้น
        """)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประเมินประสิทธิภาพ: {e}")
        st.info("อาจเกิดจากไม่พบไฟล์ข้อมูล หรือโมเดลมีปัญหา")