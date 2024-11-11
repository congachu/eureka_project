from django.shortcuts import render, redirect
import joblib
import os
import logging
import numpy as np
from django.conf import settings
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# 모델 파일 경로
model_path = os.path.join(settings.BASE_DIR, 'ai_app/spam_detector/spam_classifier_model.joblib')
logger = logging.getLogger(__name__)

# 모델 로드
def load_model():
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path, mmap_mode='r')
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    else:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

# 홈 페이지
def home(request):
    return render(request, 'home.html')

# 스팸 체크 API
def check(request):
    if request.method == 'POST':
        data = request.POST.get('email-content', '')  # 입력받은 이메일 내용

        if data:
            try:
                model = load_model()  # 모델 로드
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return render(request, 'home.html', {"error": "모델을 로드하는 중 오류가 발생했습니다. 관리자에게 문의하세요."})

            # 파이프라인 속성 복원
            if isinstance(model, Pipeline):
                # 개별 구성 요소 복원
                tfidf = model.named_steps['tfidf']
                tfidf.vocabulary_ = np.asarray(tfidf.vocabulary_).astype(object)

                classifier = model.named_steps['classifier']
                if hasattr(classifier, 'n_features_in_'):
                    classifier.n_features_in_ = len(tfidf.vocabulary_)

            # 입력 데이터를 리스트로 감싸기
            feature_vector = [data]  # 단일 입력을 리스트로 변환

            # 예측 확률
            prediction_proba = model.predict_proba(feature_vector)

            # 스팸 확률
            spam_probability = prediction_proba[0][1] * 100  # 스팸일 확률
            non_spam_probability = prediction_proba[0][0] * 100  # 비스팸일 확률

            if spam_probability > 50:  # 임계값 설정
                response = '스팸'
                probability = f"{spam_probability:.2f}"
            else:
                response = '정상'
                probability = f"{non_spam_probability:.2f}"

            return render(request, 'home.html', {"result": response, "probability": probability})

    return redirect('home')