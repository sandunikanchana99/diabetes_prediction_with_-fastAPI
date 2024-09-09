from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Split data into features and target variable
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "diabetes_model.pkl")

# Initialize the FastAPI app
app = FastAPI()

# Create a Pydantic model to define the input data format
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the trained model
model = joblib.load("diabetes_model.pkl")

@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Interpret prediction result
    if prediction[0] == 1:
        result = "You have diabetes."
    else:
        result = "You have not diabetes."
    
    return {
        "prediction": result
    }
