import os
import pandas as pd
from match_engine.tfidf_matcher import  compareResumeToJd

resume_dir = '../data/resumes'
jd_dir = '../data/job_descriptions'
output_dir_csv = '../data/scores.csv'

rows = []

for resume_file in os.listdir(resume_dir):
    resume_path = os.path.join(resume_dir,resume_file)
    for jd_file in os.listdir(jd_dir):
        jd_path = os.path.join(jd_dir,jd_file)

        score = compareResumeToJd(resume_path,jd_path)

        rows.append({
            "resume": resume_file,
            "jd_file": jd_file,
            "similarity_score": score
        })
df = pd.DataFrame(rows)
df.to_csv(output_dir_csv,index=False)

print("✅ Saved scores to:", output_dir_csv)
