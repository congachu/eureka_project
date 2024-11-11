from django.shortcuts import render, redirect
import joblib
import os
import logging
from django.conf import settings

# Define model file path
model_path = os.path.join(settings.BASE_DIR, 'ai_app/spam_detector/spam_classifier_model.joblib')
logger = logging.getLogger(__name__)

# Load the model pipeline
def load_model():
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = joblib.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    else:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

# Home page
def home(request):
    return render(request, 'home.html')

# Spam check API
def check(request):
    if request.method == 'POST':
        data = request.POST.get('email-content', '')

        if data:
            try:
                model = load_model()  # Load the pipeline model (includes vectorizer)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return render(request, 'home.html', {"error": "Error loading model, please contact the administrator."})

            # Transform input text directly using the model pipeline
            feature_vector = model.named_steps['tfidf'].transform([data])

            # Predict probability directly using predict_proba
            prediction_proba = model.named_steps['classifier'].predict_proba(feature_vector)

            spam_probability = prediction_proba[0][1] * 100  # Probability of spam
            non_spam_probability = prediction_proba[0][0] * 100  # Probability of ham

            if spam_probability > 50:  # Define threshold
                response = '스팸 메일'
                probability = f"{spam_probability:.2f}"
            else:
                response = '정상 메일'
                probability = f"{non_spam_probability:.2f}"

            return render(request, 'home.html', {"result": response, "probability": probability})

    return redirect('home')
