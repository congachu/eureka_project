import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import joblib


file_path = 'C:/Dev/python/EurekaProject/eureka/ai_app/spam_detector/spam.csv'
data = pd.read_csv(file_path, encoding='Windows-1252')
print(data.head())

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.head()
print(data.head())

data.columns = ['Label', 'Text']
X = data['Text']
y = data['Label'].map({'spam': 1, 'ham': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(X_train_tfidf, y_train)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_tfidf, y_train)

voting_model = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('rf', rf_model),
    ('gb', gb_model)
], voting='hard')  # voting='hard' 다수결로 최종 클래스 선택

# 보팅 앙상블 학습 및 평가
voting_model.fit(X_train_tfidf, y_train)
voting_pred = voting_model.predict(X_test_tfidf)

print("Voting Ensemble Accuracy:", accuracy_score(y_test, voting_pred))
print(classification_report(y_test, voting_pred, target_names=['ham', 'spam']))

joblib.dump(voting_model, 'C:/Dev/python/EurekaProject/eureka/ai_app/spam_detector/spam_classifier_model.joblib', compress=True)