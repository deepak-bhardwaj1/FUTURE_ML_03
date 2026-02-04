# main.py - Complete Resume Screening System
from data_preparation import create_sample_data
from feature_extraction import FeatureExtractor
from matching import ResumeMatcher
import pandas as pd

def main():
    print("=" * 50)
    print("RESUME SCREENING SYSTEM")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    jobs_df, resumes_df = create_sample_data()
    
    print(f"   • Jobs: {len(jobs_df)} positions")
    for _, job in jobs_df.iterrows():
        print(f"     - {job['title']}")
    
    print(f"\n   • Candidates: {len(resumes_df)} resumes")
    for _, resume in resumes_df.iterrows():
        print(f"     - {resume['name']}")
    
    # Step 2: Feature extraction
    print("\n2. Extracting features...")
    extractor = FeatureExtractor()
    
    # Prepare texts
    job_texts = list(jobs_df['description'])
    resume_texts = list(resumes_df['content'])
    
    # Fit on jobs, transform both
    job_vectors = extractor.fit_transform(job_texts)
    resume_vectors = extractor.transform(resume_texts)
    
    print(f"   • Features extracted: {len(extractor.get_features())} keywords")
    print(f"   • Sample keywords: {extractor.get_features()[:10]}")
    
    # Step 3: Matching
    print("\n3. Matching candidates to jobs...")
    matcher = ResumeMatcher()
    matcher.train(job_vectors)
    similarity = matcher.match(resume_vectors)
    results = matcher.get_top_matches(similarity, top_n=2)
    
    # Step 4: Display results
    print("\n4. Results:")
    print("=" * 50)
    
    # Group by resume
    for resume_idx, resume_row in resumes_df.iterrows():
        print(f"\n{resume_row['name']}:")
        print(f"Skills: {resume_row['content'][:50]}...")
        print("-" * 40)
        
        # Get this resume's matches
        resume_results = [r for r in results if r['resume_id'] == resume_idx]
        
        if resume_results:
            for result in resume_results:
                job_row = jobs_df.iloc[result['job_id']]
                print(f"  → {job_row['title']}: {result['score']} ({result['level']} match)")
                print(f"    Job requires: {job_row['description'][:50]}...")
        else:
            print("  No suitable matches found")
    
    # Step 5: Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("=" * 50)
    
    # For each job, show best candidate
    for job_idx, job_row in jobs_df.iterrows():
        print(f"\n{job_row['title']}:")
        
        # Get scores for this job
        job_scores = similarity[:, job_idx]
        
        if len(job_scores) > 0:
            # Find best candidate for this job
            best_resume_idx = np.argmax(job_scores)
            best_score = job_scores[best_resume_idx] * 100
            
            if best_score > 30:  # Minimum threshold
                best_candidate = resumes_df.iloc[best_resume_idx]
                print(f"  Best candidate: {best_candidate['name']} ({best_score:.1f}% match)")
                print(f"  Skills: {best_candidate['content'][:50]}...")
            else:
                print(f"  No strong candidates found (best match: {best_score:.1f}%)")
        else:
            print("  No candidates available")
    
    print("\n" + "=" * 50)
    print("SYSTEM SUMMARY:")
    print(f"• Total jobs analyzed: {len(jobs_df)}")
    print(f"• Total candidates screened: {len(resumes_df)}")
    print(f"• Total matches generated: {len(results)}")
    print("=" * 50)

if __name__ == "__main__":
    import numpy as np
    main()
