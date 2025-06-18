import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the model
model = pickle.load(open('salary_model.pkl', 'rb'))

# Streamlit UI
st.title("Salary Predictor")
st.write("Enter your years of experience and predict salary.")

# Slider input
experience = st.slider("Years of Experience", 0.0, 20.0, 2.0, 0.1)

# Predict salary
predicted_salary = model.predict(np.array([[experience]]))[0]

# Show prediction
st.subheader(f"📊 Predicted Salary: ₹{predicted_salary:,.2f}")

# Graph
df = pd.read_csv("salary_data.csv")
X = df[['YearsExperience']]
y = df['Salary']
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(experience, predicted_salary, color='green', label='Your Input', s=100)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
st.pyplot(plt)
