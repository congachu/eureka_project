import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from konlpy.tag import Okt

# with open('path_to_your_downloaded_file.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 데이터프레임으로 변환
# messages = []
# labels = []
#
# for item in data['document']:
#     messages.append(item['text'])
#     labels.append(item['metadata']['label'])
#
# df = pd.DataFrame({'text': messages, 'label': labels})
#
# # 레이블을 숫자로 변환 (spam: 1, ham: 0)
# df['label'] = df['label'].map({'spam': 1, 'ham': 0})
#
# print(df.head())
# print(f"총 데이터 수: {len(df)}")
# print(f"스팸 메시지 수: {df['label'].sum()}")
# print(f"정상 메시지 수: {len(df) - df['label'].sum()}")

# 한국어 텍스트 전처리 함수
okt = Okt()

data = pd.DataFrame({
    'text': [
        "안녕하세요, 오늘 날씨가 좋네요.",
        "특가 상품! 지금 바로 구매하세요!",
        "내일 회의 일정 알려드립니다.",
        "무료 상품권 100만원 증정! 링크를 클릭하세요!",
        "어제 보낸 이메일 확인하셨나요?",
        "당신의 계좌로 입금되었습니다. 확인해주세요."
    ],
    'spam': [0, 1, 0, 1, 0, 1]
})

def preprocess_text(text):
    # 형태소 분석 및 품사 태깅
    tokens = okt.pos(text, norm=True, stem=True)
    # 명사, 형용사, 동사만 선택
    tokens = [word for word, pos in tokens if pos in ['Noun', 'Adjective', 'Verb']]
    return ' '.join(tokens)

# 데이터 전처리
X = data['text'].apply(preprocess_text)
y = data['spam']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파이프라인 생성
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 그리드 서치를 위한 파라미터 설정
param_grid = {
    'tfidf__max_features': [1000, 2000],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

# 그리드 서치 수행
grid_search = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 최적의 모델 출력
print("최적의 파라미터:", grid_search.best_params_)

# 테스트 세트에 대한 성능 평가
y_pred = grid_search.predict(X_test)
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 교차 검증 점수
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=2)
print("\n5-fold 교차 검증 점수:", cv_scores)
print("평균 교차 검증 점수:", cv_scores.mean())

# 새로운 이메일 분류 함수
def classify_email(text):
    preprocessed_text = preprocess_text(text)
    prediction = grid_search.predict([preprocessed_text])
    score = grid_search.score(X, y)
    print(f"\n테스트 이메일의 분류 결과: {prediction}")
    return "스팸 " + str(int(score)) + "%" if prediction[0] == 1 else "정상 메일 " + str(int(score)) + "%"

# 테스트
test_email = "무료 상품권과 특별 할인 혜택을 드립니다. 지금 바로 확인하세요!"
print(f"\n테스트 이메일의 분류 결과: {classify_email(test_email)}")