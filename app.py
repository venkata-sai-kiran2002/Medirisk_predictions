from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from model import predict_risk  # Import the prediction function from model.py
from datetime import datetime
from auth import auth, token_required, is_logged_in, get_current_user
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive Agg
import matplotlib.pyplot as plt
from bson.objectid import ObjectId
from config import db

    
# Load environment variables from .env fileb
load_dotenv()
users_predictions_collection=db['predictions']

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Configure server name for proper URL generation
app.config['PREFERRED_URL_SCHEME'] = os.getenv('PREFERRED_URL_SCHEME', 'https')



# Register the auth blueprint
app.register_blueprint(auth, url_prefix='/auth')

# Ensure static directory exists
for directory in ['static', 'static/css', 'static/js', 'static/images']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Store predictions for history feature
users_predictions = []

# Context processor to make is_logged_in available in all templates
@app.context_processor
def inject_user():
    return dict(
        is_logged_in=is_logged_in(),
        current_user=get_current_user()
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not is_logged_in():
        return redirect(url_for('auth.login', next=request.url))
    
    current_user = get_current_user()
    
    if request.method == 'POST':
        # Get form data
        data = request.form
        
        # Prepare input for model
        input_data = {
            'FVC': float(data['fvc']),
            'FEV1': float(data['fev1']),
            'AGE': int(data['age']),
            'Diabetes-mellitus': 1 if data.get('diabetes') == 'yes' else 0,
            'mi-6-mo': 1 if data.get('mi') == 'yes' else 0,
            'PAD': 1 if data.get('pad') == 'yes' else 0,
            'Smoking': 1 if data.get('smoking') == 'yes' else 0,
            'Asthama': 1 if data.get('asthma') == 'yes' else 0
        }
        
        # Call the prediction function from model.py
        prediction =predict_risk(input_data)
        
        # Store prediction in history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_record = {
            'user_id': str(current_user['_id']),
            'timestamp': timestamp,
            'input': input_data,
            'result': prediction
        }
        
        # Store in local history and MongoDB
        users_predictions.append(prediction_record)
        users_predictions_collection.insert_one(prediction_record)
        
        return render_template('result.html', 
                             risk_level=prediction['risk_level'],
                             probability=prediction['probability'],
                             input_data=input_data,
                             timestamp=timestamp)
    else:
        # GET request - show the prediction form
        return render_template('predict.html')

@app.route('/history')
def history():
    if not is_logged_in():
        return redirect(url_for('auth.login', next=request.url))
    
    current_user = get_current_user()
    
    # Get user-specific predictions from MongoDB
    user_predictions = list(db.predictions.find({'user_id': str(current_user['_id'])}).sort('timestamp', -1))
    
    return render_template('history.html', predictions=user_predictions)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    
    if not is_logged_in():
        return redirect(url_for('auth.login', next=request.url))
    
    current_user = get_current_user()
    
    # Get metrics
    metrics = get_model_metrics()
    # Set image paths for template
    roc_curve = url_for('static', filename='images/roc_curve.png')
    correlation_matrix = url_for('static', filename='images/correlation_matrix.png')
    confusion_matrix = url_for('static', filename='images/confusion_matrix.png')
    
    return render_template('analysis.html',
                          username=current_user['username'] if 'username' in current_user else session.get('username', 'User'),
                          metrics=metrics,
                          roc_curve=roc_curve,
                          correlation_matrix=correlation_matrix,
                          confusion_matrix=confusion_matrix)

@app.route('/patient-analysis/<prediction_id>')
@token_required
def patient_analysis(current_user, prediction_id):
    # Find the specific prediction in MongoDB
    prediction =db.predictions.find_one({
        '_id': ObjectId(prediction_id),
        'user_id': str(current_user['_id'])
    })
    
    if not prediction:
        flash('Prediction not found or access denied', 'danger')
        return redirect(url_for('history'))
    
    # Render the patient analysis template with the prediction data
    return render_template('patient_analysis.html', 
                          prediction=prediction, 
                          username=current_user['username'] if current_user and 'username' in current_user else session.get('username', 'User'))

@app.route('/delete-history-page')
def delete_history_page():
    if not is_logged_in():
        return redirect(url_for('auth.login', next=request.url))
    
    return render_template('delete_history_page.html')

@app.route('/delete-history', methods=['POST'])
def delete_history():
    """
    Delete all prediction history for the current user
    """
    if not is_logged_in():
        return redirect(url_for('auth.login', next=request.url))
    
    current_user = get_current_user()
    
    try:
        # Delete all prediction records for the current user
        result =db.predictions.delete_many({'user_id': str(current_user['_id'])})
        
        # Also clear in-memory predictions for this user
        global users_predictions
        users_predictions = [p for p in users_predictions if p.get('user_id') != str(current_user['_id'])]
        
        if result.deleted_count > 0:
            flash(f'Successfully deleted {result.deleted_count} predictions from your history', 'success')
        else:
            flash('No prediction history found to delete', 'info')
            
    except Exception as e:
        flash(f'An error occurred while deleting history: {str(e)}', 'danger')
    
    return redirect(url_for('auth.profile'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Functions for the analysis page
def get_model_metrics():
    
    return {
        'accuracy': 85.10,
        'precision': 0.9,  # 1.00 converted to percentage
        'recall': 1.0,      # Same as sensitivity
        'f1_score': 0.92,   # 0.88 converted to percentag
        'sensitivity': 100.0,
           
    }

# This function is no longer used, as we generate images in model.py
def generate_model_visualizations(static_folder='static/images'):
    
    
    # generated directly by model.py(reference)
    os.makedirs(static_folder, exist_ok=True)
    
    # Check if model.py has been run to generate the images
    if not os.path.exists(os.path.join(static_folder, 'roc_curve.png')):
        print("Warning: Visualization images not found. Please run model.py first.")
        # Generate placeholder images with a message to run model.py
        for img_name in ['roc_curve.png', 'loss_curve.png', 'confusion_matrix.png']:
            if not os.path.exists(os.path.join(static_folder, img_name)):
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "Please run model.py to generate visualization", 
                        ha='center', va='center', fontsize=14)
                plt.axis('off')
                plt.savefig(os.path.join(static_folder, img_name))
                plt.close()
    
    return True

if __name__ == "__main__":
    print('starting Flask App.....')
    host = os.getenv('APP_HOST')
    port_str = os.getenv('APP_PORT').split()[0]  
    port = int(port_str)
    debug = os.getenv('FLASK_DEBUG').lower() == 'true'
    
    app.run(host= host, port=port, debug= debug)


