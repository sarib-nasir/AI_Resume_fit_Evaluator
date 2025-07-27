from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from match_engine.pre_process import preProcess


# Step 2: Compare resumes to JD
def compareResumeToJd(resume, jd):
    # resume = preProcess(read_file(resume_path))
    # jd = preProcess(read_file(jd_path))

    # resume = read_file(resume_path)
    # jd = read_file(jd_path)

    # Step 3: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume, jd])

    # Step 4: Cosine Similarity
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score, 3)