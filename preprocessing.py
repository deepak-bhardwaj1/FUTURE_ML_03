# preprocessing.py
import re
import string

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def test_preprocessing():
    """Test function"""
    text = "Python Developer with 5 years experience!"
    print(f"Original: {text}")
    print(f"Cleaned: {clean_text(text)}")

if __name__ == "__main__":
    test_preprocessing()
