# matching.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeMatcher:
    def __init__(self):
        """Initialize matcher"""
        self.job_vectors = None
        
    def train(self, job_vectors):
        """Store job vectors for matching"""
        self.job_vectors = job_vectors
        
    def match(self, resume_vectors):
        """Match resumes to jobs"""
        if self.job_vectors is None:
            raise ValueError("Train the matcher first!")
        
        # Calculate similarity
        similarity = cosine_similarity(resume_vectors, self.job_vectors)
        return similarity
    
    def get_top_matches(self, similarity_scores, top_n=2):
        """Get top N matches for each resume"""
        results = []
        
        for i, scores in enumerate(similarity_scores):
            # Get top N job indices
            top_indices = np.argsort(scores)[::-1][:top_n]
            
            for rank, job_idx in enumerate(top_indices):
                score = scores[job_idx] * 100  # Convert to percentage
                
                # Determine match level
                if score >= 70:
                    level = "High"
                elif score >= 40:
                    level = "Medium"
                else:
                    level = "Low"
                
                results.append({
                    'resume_id': i,
                    'job_id': job_idx,
                    'score': f"{score:.1f}%",
                    'level': level,
                    'rank': rank + 1
                })
        
        return results

def test_matching():
    """Test the matching system"""
    print("Testing Resume Matching...")
    print("=" * 40)
    
    # Sample job vectors (normally from TF-IDF)
    job_vectors = np.array([
        [1.0, 0.5, 0.3, 0.0, 0.0],  # Data Analyst job
        [0.0, 0.0, 0.0, 1.0, 0.7],  # Web Developer job
    ])
    
    # Sample resume vectors
    resume_vectors = np.array([
        [0.9, 0.6, 0.4, 0.1, 0.0],  # Alice's resume (similar to Data Analyst)
        [0.1, 0.0, 0.0, 0.8, 0.9],  # Bob's resume (similar to Web Developer)
        [0.5, 0.5, 0.5, 0.5, 0.5],  # Charlie's resume (mixed skills)
    ])
    
    # Initialize matcher
    matcher = ResumeMatcher()
    
    # Train on jobs
    matcher.train(job_vectors)
    
    # Match resumes
    similarity = matcher.match(resume_vectors)
    
    # Get results
    results = matcher.get_top_matches(similarity)
    
    # Display results
    print("\nMatching Results:")
    print("-" * 40)
    
    resume_names = ["Alice", "Bob", "Charlie"]
    job_titles = ["Data Analyst", "Web Developer"]
    
    for resume_idx in range(len(resume_names)):
        print(f"\n{resume_names[resume_idx]}:")
        # Get this resume's results
        resume_results = [r for r in results if r['resume_id'] == resume_idx]
        
        for result in resume_results:
            job_title = job_titles[result['job_id']]
            print(f"  → {job_title}: {result['score']} ({result['level']} match)")
    
    print("\n" + "=" * 40)
    print("Similarity Matrix:")
    print("Rows: Resumes, Columns: Jobs")
    print("-" * 40)
    
    # Format as percentages
    for i in range(similarity.shape[0]):
        row = [f"{score*100:.1f}%" for score in similarity[i]]
        print(f"{resume_names[i]}: {row}")
    
    return matcher

if __name__ == "__main__":
    test_matching()
