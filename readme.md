

# 📄 **Project Report**

## 🔰 Project Title:

**Salary Prediction Using Linear Regression**

---

## 👨‍🎓 Submitted By:

**Aditya Kumar Gupta**
**B.Tech – Computer Science Engineering (2nd Year)**
**Institute: REC**

---

## 🎯 Project Objective:

To build a **machine learning model** that predicts an employee’s salary based on their **years of experience**, and to deploy it using a simple **Streamlit web application**.

---

## 🔍 Problem Statement:

In many industries, salary estimation is important for both employers and employees. Manual analysis can be biased and time-consuming. This project aims to automate the process using **Simple Linear Regression**, making the prediction process quick and accurate.

---

## 🧱 Tools & Technologies Used:

| Module           | Use / Purpose                                     |
| ---------------- | ------------------------------------------------- |
| **Python**       | Programming language for scripting logic          |
| **pandas**       | Data handling (CSV loading, preprocessing)        |
| **numpy**        | Numerical operations                              |
| **scikit-learn** | Machine Learning model (Linear Regression)        |
| **matplotlib**   | Data visualization and residual analysis          |
| **pickle**       | Save and load trained ML model                    |
| **Streamlit**    | Build interactive UI (slider, graph, predictions) |

---

## 🔧 Tools Required

* Python
* pandas, numpy, matplotlib, scikit-learn
* Streamlit

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

---

## 📂 Dataset Description:

* **File Name:** `salary_data.csv` 
* **Source:** Kaggle 
* **Columns:**

    * `YearsExperience` (float): Number of years of work experience
    * `Salary` (float): Actual salary in ₹
* **Records:** 20–30 sample rows used for training and testing

---

## 🔧 Module-wise Explanation:

### 1. **Data Preprocessing Module**

* **Library Used:** `pandas`
* **Task:** Load CSV dataset, clean if necessary, prepare for training.

```python
df = pd.read_csv("salary_data.csv")
```

---

### 2. **Model Training Module**

* **Library Used:** `scikit-learn`
* **Task:** Train Simple Linear Regression model with input (X) as YearsExperience and output (y) as Salary.
* **Output:** `salary_model.pkl` (serialized model)

```python
model = LinearRegression()
model.fit(X, y)
```

---

### 3. **Model Saving Module**

* **Library Used:** `pickle`
* **Task:** Save the trained model so that it can be used in UI without retraining.

```python
pickle.dump(model, open('salary_model.pkl', 'wb'))
```

---

### 4. **Residual Analysis Module**

* **Library Used:** `matplotlib`
* **Task:** Check accuracy and performance by visualizing residuals (actual - predicted).

```python
residuals = y - y_pred
```

---

### 5. **Streamlit Web App Module**

* **Library Used:** `streamlit`
* **Task:** Build an interactive user interface:

    * Slider to select Years of Experience
    * Show predicted salary
    * Show graph of original data + regression line + predicted point

```python
experience = st.slider("Years of Experience", 0.0, 20.0, 2.0, 0.1)
predicted_salary = model.predict([[experience]])
```

---

## 📊 Output Sample (UI):

* **Slider:** Select years of experience (e.g., 3.5 years)
* **Output:** Predicted salary shown dynamically
* **Graph:** Shows:

    * Blue dots = original data
    * Red line = regression line
    * Green point = your selected experience

---

## ✅ Project Outcome:

* A fully working web application that predicts salary using ML
* Clean and simple UI for real-time predictions
* Core ML concepts like model training, saving, loading, and prediction applied

---

## 🧠 Learnings:

* Basics of Linear Regression
* Handling datasets in pandas
* Training and saving models with scikit-learn & pickle
* UI design with Streamlit
* Deploying local ML models in real-world interfaces

---

## ▶️ Run the App

In terminal:

```bash
streamlit run app.py
```

It will open in browser with slider input, prediction, and graph.

---

## 📁 Project Folder Structure:

```
salary-prediction/
├── app.py                 # Streamlit app
├── model.py               # Model training script
├── salary_data.csv        # Dataset
├── salary_model.pkl       # Trained ML model
└── residual_plot.png      # Residual plot image
```

---

## 🙏 Acknowledgements:

I would like to thank my faculty and classmates for guiding and supporting me in completing this project successfully.

