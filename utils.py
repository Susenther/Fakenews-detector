import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    """
    Downloads the necessary NLTK datasets.
    Updates: Added 'punkt_tab' which is required for newer NLTK versions.
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt')

    # --- NEW REQUIREMENT ---
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download('punkt_tab')
    # -----------------------

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet lemmatizer...")
        nltk.download('wordnet')

# Run the download check when this module is first imported
download_nltk_data()

# Initialize the lemmatizer and stopwords list for efficiency
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses raw text for the classical ML model.
    """
    # 1. Handle non-string inputs (like 'nan' or None)
    if not isinstance(text, str):
        return ""

    # 2. Lowercase text
    text = text.lower()
    
    # 3. Remove punctuation and numbers
    # Keep only alphabetic characters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize text
    # This step requires 'punkt_tab' in newer NLTK versions
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 7. Join tokens back into a string
    return ' '.join(tokens)