# 🎓 Student Success & At-Risk Prediction Model

This project predicts whether a student is **at risk** of underperforming or **likely to succeed** academically based on their profile and performance indicators.  
It uses **XGBoost** as the primary machine learning model, trained on student success datasets.  

---

## 📂 Project Structure
```
├── Student Success & At-Risk Prediction Model.ipynb   # Jupyter Notebook with full workflow
├── student_success_dataset.csv.xls                    # Dataset used for model training
├── student_at_risk_predictor_xgb.pkl                  # Trained XGBoost model (saved)
└── README.md                                          # Project documentation
```

---

## 🚀 Features
- Data preprocessing (cleaning, encoding, scaling).  
- Exploratory Data Analysis (EDA).  
- Training with **XGBoost Classifier**.  
- Performance evaluation using accuracy, precision, recall, F1-score.  
- Exporting trained model for reuse.  
- Predicting student success probability for new inputs.  

---

## 🛠️ Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

**Key libraries:**
- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- matplotlib / seaborn (for visualization)  
- joblib (for saving/loading model)  

---

## 📊 Dataset
The dataset (`student_success_dataset.csv.xls`) contains features like:  
- **Demographics** (age, gender, etc.)  
- **Academic performance** (grades, test scores)  
- **Attendance / participation**  
- **Socioeconomic indicators**  

Target variable:  
- `Success` → 1 (Successful), 0 (At-Risk).  

---

## 📓 Usage
### 1. Run the Notebook
Open and execute **Student Success & At-Risk Prediction Model.ipynb** in Jupyter Notebook / JupyterLab.

### 2. Load the Pretrained Model
```python
import joblib

# Load model
model = joblib.load("student_at_risk_predictor_xgb.pkl")

# Example prediction
sample = [[...]]  # Replace with student features
prediction = model.predict(sample)
print("At-Risk" if prediction[0] == 0 else "Successful")
```

### 3. Train Your Own Model
Modify the notebook to retrain the model with new data.  

---

## 📈 Model Performance
- **Model**: XGBoost  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score  
- Achieved high F1-score, making it suitable for imbalanced datasets where predicting "At-Risk" cases is critical.  

---

## 🔮 Applications
- Early intervention for at-risk students.  
- Academic advising and personalized learning plans.  
- Institutional reporting and student success strategies.  

---

## 👨‍💻 Author
Developed by **[Saksham Tapadia]**  
📧 Contact: [sakshamtapadia10@gmail.com]  
