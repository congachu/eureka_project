from django.shortcuts import render, redirect
import joblib
import os
import logging
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from .models import EmailCheck

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

            email_check = EmailCheck.objects.create(
                user=request.user if request.user.is_authenticated else None,  # 로그인된 유저가 있다면 저장
                email_content=data,
                result=response,
                probability=probability
            )

            return render(request, 'home.html', {"result": response, "probability": probability})

    return redirect('home')


@require_http_methods(["GET", "POST"])
def login(request):
    # If user is already logged in, redirect to home
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        # Get username and password from POST data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Login the user and create a session
            auth_login(request, user)

            # Optional: Set session expiry (example: 1 week)
            request.session.set_expiry(604800)  # 7 days in seconds

            # Add a success message
            messages.success(request, f'{username}님 환영합니다!')

            # Redirect to home page
            return redirect('home')
        else:
            # Add an error message for failed login
            messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')

    # Render login page for GET request
    return render(request, 'login.html')


@require_http_methods(["POST", "GET"])
def register(request):
    # If user is already logged in, redirect to home
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # Basic validation
        if not username or not password:
            messages.error(request, '아이디와 비밀번호를 모두 입력해주세요.')
            return render(request, 'register.html')

        if password != confirm_password:
            messages.error(request, '비밀번호가 일치하지 않습니다.')
            return render(request, 'register.html')

        User = get_user_model()
        if User.objects.filter(username=username).exists():
            messages.error(request, '이미 존재하는 아이디입니다.')
            return render(request, 'register.html')

        # Create new user
        try:
            user = User.objects.create_user(username=username, password=password)

            # Automatically log in the user after registration
            auth_login(request, user)

            messages.success(request, '회원가입이 성공적으로 완료되었습니다.')
            return redirect('home')

        except Exception as e:
            messages.error(request, f'회원가입 중 오류가 발생했습니다: {str(e)}')

    return render(request, 'registration.html')


@login_required
def logout(request):
    # Log out the user and destroy the session
    auth_logout(request)
    messages.success(request, '로그아웃 되었습니다.')
    return redirect('home')


@login_required
def profile(request):
    # 로그인한 사용자
    user = request.user

    # 해당 사용자의 점검 기록 가져오기
    user_checks = EmailCheck.objects.filter(user=user).order_by('-created_at')

    # 사용자 정보 및 점검 기록을 템플릿에 전달
    return render(request, 'profile.html', {'user': user, 'user_checks': user_checks})
