from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib
import os

# Initialize the FastAPI app
app = FastAPI()

# Define the Pydantic model to validate the input data
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the dataset (make sure the 'diabetes.csv' file is in the correct path)
data_path = "diabetes.csv"
if os.path.exists(data_path):
    data = pd.read_csv(data_path)

    # Split data into features and target variable
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model to a file
    model_filename = "diabetes_model.pkl"
    joblib.dump(model, model_filename)
else:
    print(f"Dataset {data_path} not found.")

# Load the trained model (ensure that the model is loaded before making predictions)
model_filename = "diabetes_model.pkl"
if os.path.exists(model_filename):
    model = joblib.load(model_filename)
else:
    raise FileNotFoundError(f"Model file {model_filename} not found.")

# Define the predict route
@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_df)
        
        # Interpret the result
        if prediction[0] == 1:
            result = "You have diabetes."
        else:
            result = "You do not have diabetes."
        
        return {
            "prediction": result
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }

# Default root endpoint for testing
@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Prediction API"}
