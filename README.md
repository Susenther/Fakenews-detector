# Fakenews-detector

A Machine Learning application built with Python and Streamlit that classifies news articles as Real or Fake using Natural Language Processing (NLP).

🚀 Features

Instant Analysis: Paste any news article text to get an immediate credibility check.

Clean UI: User-friendly interface built with Streamlit.

High Accuracy: Uses TF-IDF vectorization and Logistic Regression trained on the ISOT Fake News Dataset.

Visual Confidence: Displays results with clear "Real" (Green) or "Fake" (Red) indicators.

🛠️ Tech Stack

Language: Python 3.x

Frontend: Streamlit

Machine Learning: Scikit-learn (Logistic Regression, TF-IDF)

Data Processing: Pandas, NLTK

Serialization: Joblib

📂 Project Structure

fakenews-app/
├── data/                   # Folder for True.csv and Fake.csv (dataset)
├── models/                 # Saved models (tfidf_vectorizer.pkl, lr_model.pkl)
├── app.py                  # Streamlit frontend application
├── train_classical.py      # Script to train and save the model
├── utils.py                # Helper functions (text preprocessing)
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation


⚙️ Installation

Clone the repository:

git clone [https://github.com/Susenther/Fake-news-detector.git](https://github.com/Susenther/Fake-news-detector.git)
cd Fake-news-detector


Install dependencies:

pip install -r requirements.txt


Download NLTK data (if prompted):
The app will automatically try to download necessary NLTK data, but if you see an error, run:

import nltk
nltk.download('punkt')
nltk.download('wordnet')


🏃‍♂️ Usage

Train the Model:
Before running the app, you must train the model once to generate the model files in the models/ folder.

python train_classical.py


Run the App:
Launch the Streamlit interface.

streamlit run app.py


View:
Open the URL provided in the terminal (usually http://localhost:8501) in your browser.

📊 Dataset

This project uses the ISOT Fake News Dataset, consisting of thousands of real and fake news articles.
