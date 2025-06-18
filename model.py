import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv('./salary_data.csv')
X = df[['YearsExperience']]
y = df['Salary']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
pickle.dump(model, open('salary_model.pkl', 'wb'))


