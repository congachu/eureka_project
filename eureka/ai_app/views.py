import random

from django.http import JsonResponse
from django.shortcuts import render, redirect
import joblib
import os
import logging
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
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
            return render(request, 'registration.html')

        if password != confirm_password:
            messages.error(request, '비밀번호가 일치하지 않습니다.')
            return render(request, 'registration.html')

        User = get_user_model()
        if User.objects.filter(username=username).exists():
            messages.error(request, '이미 존재하는 아이디입니다.')
            return render(request, 'registration.html')

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
    user_checks_list = EmailCheck.objects.filter(user=user).order_by('-created_at')

    # 페이지네이션 설정 (한 페이지에 10개의 레코드)
    paginator = Paginator(user_checks_list, 10)

    # 현재 페이지 번호 가져오기
    page = request.GET.get('page', 1)

    try:
        user_checks = paginator.page(page)
    except PageNotAnInteger:
        # 페이지 번호가 정수가 아닌 경우 첫 페이지 보여주기
        user_checks = paginator.page(1)
    except EmptyPage:
        # 페이지 범위를 벗어난 경우 마지막 페이지 보여주기
        user_checks = paginator.page(paginator.num_pages)

    # 사용자 정보 및 점검 기록을 템플릿에 전달
    return render(request, 'profile.html', {
        'user': user,
        'user_checks': user_checks
    })


def recommend_email(request):
    recommendations = [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "Ok lar... Joking wif u oni...",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "U dun say so early hor... U c already then say...",
        "Nah I don't think he goes to usf, he lives around here though",
        "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 鶯1.50 to rcv",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune",
        "WINNER!! As a valued network customer you have been selected to receivea 鶯900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
    ]
    recommendation = random.choice(recommendations)

    # JSON 응답 반환
    return JsonResponse({"recommendation": recommendation})