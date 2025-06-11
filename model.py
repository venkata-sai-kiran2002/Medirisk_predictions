# model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix


# Load and preprocess data
def preprocess_data(data):
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.Categorical(data[col]).codes + 1
    data.fillna(data.median(), inplace=True)
    return data

data = pd.read_csv("data/ThoracicCancerSurgery.csv")
data = preprocess_data(data)
print(data.head())

print(data.info())

print(data.isnull().sum())
for label, content in data.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)

print(data["Risk1Yr"].value_counts())

df=pd.crosstab(data['AGE'],data['Risk1Yr'])

# Filter data based on DRK_YN values
R_1 = data[data['Risk1Yr'] == 1]
R_2 = data[data['Risk1Yr'] == 2]


# Plotting
plt.figure(figsize=(10, 6))
plt.hist(R_1['AGE'], bins=20, alpha=0.5, label='Died within 1 yr')
plt.hist(R_2['AGE'], bins=20, alpha=0.5, label="Survived")

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Correlation analysis
corr_matrix=data.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,
              annot=True,
              linewidth=0.5,
              fmt=".2f",
              cmap="YlGnBu")

correlation_threshold = 0.1
features = []
for feature in corr_matrix.columns:
    if feature != 'Risk1Yr' and (corr_matrix[feature]['Risk1Yr'] > correlation_threshold or
                                corr_matrix[feature]['Risk1Yr'] < -correlation_threshold):
        features.append(feature)
print("Features with correlation above threshold:", features)

plt.title('correlation_matrix')
plt.savefig("static/images/correlation_matrix.png")
plt.show()
print("Correlation matrix image saved to static/images/correlation_matrix.png")



features = ['FVC', 'FEV1', 'Asthama', 'Smoking', 'PAD', 'mi-6-mo', 'Diabetes-mellitus', 'AGE']
X = data[features]
y = data['Risk1Yr'].astype(int)



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize and train a logistic regression model and SVC
lgr_model = LogisticRegression(max_iter=1000)  
lgr_model.fit(X_train_scaled, y_train)
y_pred_lgr = lgr_model.predict(X_test_scaled)
accuracy_lgr = accuracy_score(y_test, y_pred_lgr)
print("Logistic Regression Accuracy:", accuracy_lgr)
print("precision_score: ", precision_score(y_test, y_pred_lgr))
print("f1_score: ", f1_score(y_test, y_pred_lgr))
print("recall_score: ", recall_score(y_test, y_pred_lgr))

svc_model = svm.LinearSVC(max_iter=1000)
svc_model.fit(X_train_scaled, y_train)
# Predict the target variable on the test set
y_pred_svc = svc_model.predict(X_test_scaled)
# Calculate the accuracy of the mode
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("SVM Accuracy:", accuracy_svc)
print("precision_score: ", precision_score(y_test, y_pred_svc))
print("f1_score: ", f1_score(y_test, y_pred_svc))
print("recall_score: ", recall_score(y_test, y_pred_svc))

rf_model=RandomForestClassifier()
rf_model.fit(X_train_scaled,y_train)
y_pred=rf_model.predict(X_test_scaled)
accuracy_rf=accuracy_score(y_test,y_pred)
print("Random Forest Accuracy:", accuracy_rf)
print("precision_score: ", precision_score(y_test, y_pred))
print("f1_score: ", f1_score(y_test, y_pred))
print("recall_score: ", recall_score(y_test, y_pred))

#We can Hyper tune the Logistic regression model
log_reg_grid={"C":np.logspace(-4,4,30),
             "solver":["liblinear"],
             "class_weight": ['balanced']}
#Logictic Regression
lreg_gs=GridSearchCV(LogisticRegression(random_state=42),
                     log_reg_grid,
                      cv=5,
                    verbose=True,
                    )
lreg_gs.fit(X_train_scaled,y_train)
# Evaluate the best model
print("Best parameters from Grid Search:", lreg_gs.best_params_)


pred=lreg_gs.predict(X_test_scaled)
y_preds=lreg_gs.predict(X_test_scaled)

#Roc curve and confusion matrix


# Confusion matrix
cn=confusion_matrix(y_test,y_preds)
def plot_conf_mat(y_test,y_preds):
    fig, ax =plt.subplots()
    ax=sns.heatmap(confusion_matrix(y_test,y_preds),
                  annot=True,
                  cbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("static/images/confusion_matrix.png")
    print("Confusion matrix image saved to static/images/confusion_matrix.png")
    plt.close()
    plt.show()
# plt.show() 

plot_conf_mat(y_test, y_preds)
    


RocCurveDisplay.from_estimator(lreg_gs, X_test_scaled, y_test).plot()
plt.title("ROC Curve")
plt.savefig("static/images/roc_curve.png")
plt.close()
plt.show()



# Save the best model and scaler
joblib.dump(lreg_gs, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved.")

# Exportable function for Flask to use
def predict_risk(patient_data, threshold=0.5):
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    required_features = ['FVC', 'FEV1', 'Asthama', 'Smoking', 'PAD', 'mi-6-mo', 'Diabetes-mellitus', 'AGE']
    input_data = pd.DataFrame([[float(patient_data[f]) for f in required_features]], columns=required_features)
    input_scaled = scaler.transform(input_data)

    proba = model.predict_proba(input_scaled)[0]
    high_risk_prob = proba[model.classes_.tolist().index(1)]
    risk_percent = round(high_risk_prob * 100, 2)

    if high_risk_prob >= threshold:
        risk_level = "High Risk"
        confidence = "High" if high_risk_prob >= threshold + 0.2 else "Medium"
    else:
        risk_level = "Low Risk"
        confidence = "High" if high_risk_prob <= threshold - 0.2 else "Medium"

    return {
        'risk_level': risk_level,
        'probability': risk_percent,
        'confidence': confidence,
        'features': patient_data
    }
if __name__ == "__main__":
    # Example usage
    example_patient = {
        'FVC': 3.0,
        'FEV1': 2.5,
        'Asthama': 0,
        'Smoking': 1,
        'PAD': 0,
        'mi-6-mo': 0,
        'Diabetes-mellitus': 1,
        'AGE': 65
    }
    result = predict_risk(example_patient)
    print(result)
# This code is designed to be used in a Flask application, where it can be imported and used to make predictions based on user input.
# The function `predict_risk` can be called with a dictionary of patient data to get the risk assessment.
