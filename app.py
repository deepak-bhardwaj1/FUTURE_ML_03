# app.py - Simple Interactive App
import pandas as pd
from data_preparation import create_sample_data
from feature_extraction import FeatureExtractor
from matching import ResumeMatcher
import numpy as np

def interactive_app():
    print("\n" + "=" * 60)
    print("RESUME SCREENING SYSTEM - INTERACTIVE MODE")
    print("=" * 60)
    
    # Load data
    jobs_df, resumes_df = create_sample_data()
    
    # Initialize system
    extractor = FeatureExtractor()
    job_texts = list(jobs_df['description'])
    resume_texts = list(resumes_df['content'])
    
    # Train
    job_vectors = extractor.fit_transform(job_texts)
    resume_vectors = extractor.transform(resume_texts)
    
    matcher = ResumeMatcher()
    matcher.train(job_vectors)
    similarity = matcher.match(resume_vectors)
    
    while True:
        print("\nOptions:")
        print("1. View all matches")
        print("2. Find best candidate for a job")
        print("3. Find best job for a candidate")
        print("4. Add new job description")
        print("5. Add new resume")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == "1":
            # View all matches
            results = matcher.get_top_matches(similarity)
            for result in results:
                resume_name = resumes_df.iloc[result['resume_id']]['name']
                job_title = jobs_df.iloc[result['job_id']]['title']
                print(f"{resume_name} → {job_title}: {result['score']}")
        
        elif choice == "2":
            # Best candidate for job
            print("\nAvailable Jobs:")
            for idx, job in jobs_df.iterrows():
                print(f"{idx+1}. {job['title']}")
            
            job_choice = int(input("Select job (1-3): ")) - 1
            job_scores = similarity[:, job_choice]
            best_idx = np.argmax(job_scores)
            best_score = job_scores[best_idx] * 100
            
            print(f"\nBest candidate: {resumes_df.iloc[best_idx]['name']}")
            print(f"Match score: {best_score:.1f}%")
        
        elif choice == "3":
            # Best job for candidate
            print("\nAvailable Candidates:")
            for idx, resume in resumes_df.iterrows():
                print(f"{idx+1}. {resume['name']}")
            
            resume_choice = int(input("Select candidate (1-3): ")) - 1
            resume_scores = similarity[resume_choice, :]
            best_idx = np.argmax(resume_scores)
            best_score = resume_scores[best_idx] * 100
            
            print(f"\nBest job: {jobs_df.iloc[best_idx]['title']}")
            print(f"Match score: {best_score:.1f}%")
        
        elif choice == "4":
            print("\nFeature: Add new job (coming soon)")
        
        elif choice == "5":
            print("\nFeature: Add new resume (coming soon)")
        
        elif choice == "6":
            print("\nThank you for using Resume Screening System!")
            break
        
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    interactive_app()
