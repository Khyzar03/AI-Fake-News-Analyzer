from django.shortcuts import render
import joblib
import os
from config.settings import BASE_DIR

# Load models when the server starts
try:
    model_path = os.path.join(BASE_DIR, 'ml_pipeline', 'model.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'ml_pipeline', 'vectorizer.pkl')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None

def index(request):
    prediction = None
    news_text = ""
    
    if request.method == "POST":
        news_text = request.POST.get('news_text', '')
        
        if news_text and model and vectorizer:
            # Predict
            vectorized_input = vectorizer.transform([news_text])
            result = model.predict(vectorized_input)[0] # 'REAL' or 'FAKE'
            
            # For UI aesthetics, let's also pass the label
            prediction = {
                'label': result,
                'is_fake': result == 'FAKE',
                'is_real': result == 'REAL'
            }
            
    context = {
        'prediction': prediction,
        'news_text': news_text
    }
    return render(request, 'detector/index.html', context)
