import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# Load data
file_path = 'C:/Dev/python/EurekaProject/eureka/ai_app/spam_detector/spam.csv'
data = pd.read_csv(file_path, encoding='Windows-1252')
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.columns = ['Label', 'Text']

# Prepare features and labels
X = data['Text']
y = data['Label'].map({'spam': 1, 'ham': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual classifiers
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Create a VotingClassifier with soft voting
voting_model = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('rf', rf_model),
    ('gb', gb_model)
], voting='soft')

# Create a pipeline with TfidfVectorizer and the VotingClassifier
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', voting_model)
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Evaluate the model
X_test_tfidf = model_pipeline.named_steps['tfidf'].transform(X_test)
voting_pred = model_pipeline.named_steps['classifier'].predict(X_test_tfidf)
print("Voting Ensemble Accuracy:", accuracy_score(y_test, voting_pred))
print(classification_report(y_test, voting_pred, target_names=['ham', 'spam']))

# Save the entire pipeline
joblib.dump(model_pipeline, 'C:/Dev/python/EurekaProject/eureka/ai_app/spam_detector/spam_classifier_model.joblib', compress=True)
