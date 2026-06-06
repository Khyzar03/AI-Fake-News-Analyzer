# AI Fake News Analyzer

An end-to-end Machine Learning web application built with Django that detects and prevents the spread of misinformation. 

This project analyzes news text using advanced Natural Language Processing (NLP) techniques to determine the probability of the content being factual or artificially sensationalized/fake.

## ✨ Features
*   **Machine Learning Pipeline:** Trained on a comprehensive dataset of 40,000+ real and fake news articles.
*   **High Accuracy:** Achieves **99.47% accuracy** using a `PassiveAggressiveClassifier` model.
*   **Text Processing:** Utilizes a custom `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to extract meaningful linguistic patterns.
*   **RESTful Backend:** Built on the robust Django framework, efficiently loading and caching serialized `.pkl` models for real-time inference.
*   **Premium Glassmorphism UI:** A sleek, dark-mode, responsive user interface utilizing custom CSS, dynamic gradients, and animated orbs for a modern feel.

## 🛠️ Technologies Used
### Backend (AI System)
*   **Language:** Python 3
*   **Machine Learning:** Scikit-Learn (scikit-learn)
*   **Data Processing:** Pandas, NumPy
*   **NLP Techniques:** TF-IDF (Term Frequency-Inverse Document Frequency)
*   **Model Architecture:** Passive Aggressive Classifier (Optimized for massive text streams)
*   **Model Serialization:** Joblib

### Backend (Web Server)
*   **Framework:** Django 6
*   **Architecture:** MVT (Model-View-Template)

### Frontend
*   **Markup:** HTML5 + Django Template Language
*   **Styling:** Custom CSS3 (Glassmorphism, CSS Animations, CSS Variables)
*   **Typography:** Google Fonts (Inter)

## 🚀 How It Works
1.  **Data Ingestion:** The user pastes a news excerpt into the frontend text area.
2.  **Vectorization:** The Django view (`views.py`) intercepts the POST request and passes the raw text into the pre-trained `TfidfVectorizer`. This converts the string into a numerical array of term frequencies.
3.  **Classification:** The sparse array is passed to the `PassiveAggressiveClassifier`, which evaluates the linguistic markers against its trained weights to determine a `REAL` or `FAKE` label.
4.  **UI Feedback:** The backend routes the contextual result back to the template, dynamically updating the UI to render a green or red stylized response card indicating the text's authenticity.

## 💻 Local Setup
1.  Clone the repository.
2.  Create a virtual environment: `python -m venv venv`
3.  Activate the environment: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4.  Install dependencies: `pip install django scikit-learn pandas numpy joblib`
5.  Run the application: `python manage.py runserver`
6.  Visit `http://127.0.0.1:8000` in your web browser.
