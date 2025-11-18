import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import our preprocessing function from utils.py
try:
    from utils import preprocess_text
except ImportError:
    print("Error: 'utils.py' not found. Make sure it's in the same directory.")
    exit()

print("Starting the classical model training pipeline...")

# --- 1. Load and Prepare Data ---
print("Loading data...")
try:
    # Using the filenames that worked for you earlier
    df_true = pd.read_csv("data/True.csv")
    df_fake = pd.read_csv("data/Fake.csv")
except FileNotFoundError:
    print("Error: 'True.csv' or 'Fake.csv' not found in 'data/' directory.")
    exit()
except PermissionError:
    print("Error: Permission denied. Close Excel or use the 'Unlocked' copies if you made them.")
    exit()

# Create labels and merge
df_true['label'] = 1
df_fake['label'] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)

# Combine title and text
df['full_text'] = df['title'] + ' ' + df['text']

# Select final features and shuffle
df = df[['full_text', 'label']]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data loading and preparation complete.")

# --- 2. Apply NLP Preprocessing ---
print("Preprocessing text data... (This may take a few minutes)...")
# Apply the function from utils.py
df.dropna(subset=['full_text'], inplace=True) # Ensure no NaN values
X = df['full_text'].apply(preprocess_text)
y = df['label']
print("Text preprocessing complete.")

# --- 3. Train-Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Feature Extraction (TF-IDF) - Step 2.2 ---
print("Starting TF-IDF vectorization...")
# We limit features to the top 5000 and include 1- and 2-word phrases
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Only transform the test data (using the vocab from training)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("TF-IDF vectorization complete.")

# --- 5. Model Training (Logistic Regression) - Step 2.3 ---
print("Training Logistic Regression model...")
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 6. Model Evaluation ---
print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
# Use \n for new lines instead of actual line breaks
print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake','Real']))

# --- 7. Model Serialization (Saving) ---
print("Serializing models to disk...")
# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

# Save the trained Logistic Regression Model
joblib.dump(model, 'models/lr_model.pkl')

print("\n--- Pipeline Finished Successfully! ---")
print("Models 'tfidf_vectorizer.pkl' and 'lr_model.pkl' are saved in the 'models/' directory.")