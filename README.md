<<<<<<< HEAD
# Thoracic-Cancer-Surgery-predictions
=======
# Thoracic Cancer Surgery Risk Prediction

A web application that predicts post-operative life expectancy for thoracic cancer surgery patients using machine learning.

## Features

- Patient data input through a user-friendly interface
- Risk assessment with probability estimation and risk levels
- Detailed visualization of results
- History tracking of previous predictions and Analysis of that particular individual patient's data
- RESTful API for integration with other systems(Endpoints)

## Technology Stack

- **Backend**: Flask framework (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Database**: MongoDB (for storing prediction history and user Authentication)
- **Machine Learning**: Scikit-learn, Hyper tunning with Logistic Regression model for the predictions and comparing the each other with Random Forest and SVm for the better performance of the model to get perfect prediction results.

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/thoracic-surgery-prediction.git
cd thoracic-surgery-prediction
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:
   - Run the model.py  (To load and save the model training data and Getting the results of the model)   
   - Run the app.py    (To run the Flask application)
4. Open your web browser and navigate to `http://127.0.0.1:5000/`  (To access the web application in Local Host)

## Data Input Parameters

The model requires the following patient parameters:

- **FVC** (Forced Vital Capacity): The total volume of air that can be exhaled during a forced breath
- **FEV1** (Forced Expiratory Volume in 1 second): The volume of air that can be exhaled in the first second of a forced breath
- **Age**: Patient's age
- **Presence of comorbidities**:
  - Asthma
  - Smoking
  - PAD (Peripheral Arterial Disease)
  - MI in last 6 months (Myocardial Infarction)
  - Diabetes Mellitus

## API Usage

The application provides a RESTful API for programmatic access:

```
POST /api/predict
Content-Type: application/json

{
  "fvc": 2.5,
  "fev1": 1.8,
  "age": 65,
  "diabetes": true,
  "mi": false,
  "pad": false,
  "smoking": true,
  "asthma": false
}
```

Response:
```
{
  "status": "success",
  "prediction": {
    "risk_level": "High Risk",
    "probability": 78.5
  },
  "timestamp": "2024-05-21 14:32:45"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thoracic Surgery Dataset
- Medical professionals who provided domain expertise 
>>>>>>> 56fe0f76 (first commit)
