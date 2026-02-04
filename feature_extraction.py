# feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self):
        """Initialize TF-IDF Vectorizer"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=50
        )
    
    def fit_transform(self, texts):
        """Fit and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform new texts"""
        return self.vectorizer.transform(texts)
    
    def get_features(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()

def test_feature_extraction():
    """Test function"""
    print("Testing Feature Extraction...")
    
    # Sample data
    job_texts = [
        "Python SQL Excel data analysis",
        "HTML CSS JavaScript web development"
    ]
    
    # Initialize
    extractor = FeatureExtractor()
    
    # Fit and transform
    tfidf_matrix = extractor.fit_transform(job_texts)
    
    print(f"Matrix shape: {tfidf_matrix.shape}")
    print(f"Features: {extractor.get_features()}")
    
    return extractor

if __name__ == "__main__":
    test_feature_extraction()
