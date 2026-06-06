import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Step 1: Loading Kaggle Dataset...")
output_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(output_dir, "data")

try:
    # Load the CSV files
    # The standard Kaggle Fake News dataset has 'Fake.csv' and 'True.csv'
    fake_df = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    true_df = pd.read_csv(os.path.join(data_dir, "True.csv"))
    
    # Add labels
    fake_df['label'] = 'FAKE'
    true_df['label'] = 'REAL' # Note: the web app expects 'REAL' and 'FAKE'
    
    # Combine the dataframes
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Kaggle dataset usually has 'title' and 'text'. We can combine them for better context
    # If your dataset only has 'text', just use that instead
    if 'title' in df.columns:
        df['combined_text'] = df['title'] + " " + df['text']
    else:
        df['combined_text'] = df['text']
        
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
except Exception as e:
    print(f"Error loading CSV files. Please ensure 'Fake.csv' and 'True.csv' are in {data_dir}")
    print(f"Error details: {e}")
    exit(1)

print("Step 2: Splitting dataset...")
x_train, x_test, y_train, y_test = train_test_split(df['combined_text'], df['label'], test_size=0.25, random_state=7)

print("Step 3: Initializing TfidfVectorizer...")
# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

print("Step 4: Training PassiveAggressiveClassifier...")
# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {round(score*100,2)}%')

print("Step 5: Saving the model and vectorizer...")
joblib.dump(pac, os.path.join(output_dir, 'model.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))

print("Completed successfully! model.pkl and vectorizer.pkl have been created inside ml_pipeline/")
