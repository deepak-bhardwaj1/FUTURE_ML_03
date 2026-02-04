# data_preparation.py
import pandas as pd

def create_sample_data():
    """Create sample job descriptions and resumes"""
    
    job_descriptions = [
        {"id": 1, "title": "Data Analyst", 
         "description": "Proficient in Python, SQL, Excel, data visualization, statistical analysis"},
        {"id": 2, "title": "Frontend Developer", 
         "description": "HTML, CSS, JavaScript, React, responsive design, UI/UX"},
        {"id": 3, "title": "Marketing Manager", 
         "description": "digital marketing, SEO, social media, content strategy, analytics"}
    ]
    
    resumes = [
        {"id": 101, "name": "Alice Sharma", 
         "content": "Experienced in Python programming, SQL databases, Excel analysis, data visualization"},
        {"id": 102, "name": "Bob Singh", 
         "content": "Skilled in HTML, CSS, JavaScript and React framework for web development"},
        {"id": 103, "name": "Charlie Gupta", 
         "content": "Expert in digital marketing strategies, SEO optimization, social media campaigns"}
    ]
    
    return pd.DataFrame(job_descriptions), pd.DataFrame(resumes)

def test():
    """Test function"""
    jobs_df, resumes_df = create_sample_data()
    print("Sample data created!")
    print("\nJobs:", jobs_df[['id', 'title']].to_string())
    print("\nCandidates:", resumes_df[['id', 'name']].to_string())

if __name__ == "__main__":
    test()
