import streamlit as st
from match_engine.pre_process import extractData
from match_engine.skills_extraction import extract_keywords
from match_engine.tfidf_matcher import compareResumeToJd

st.set_page_config(page_title="Job Compatibility Checker", layout="wide")  # <-- WIDE layout
st.title("📄 Job Compatibility Checker")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Your CV")
    cv_file = st.file_uploader("Upload your CV", type=["pdf", "docx", "txt"], key="cv_file")
    cv_text = st.text_area("Or paste your CV here", height=200, key="cv_text")

with col2:
    st.markdown("### Job Description (JD)")
    jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_file")
    jd_text = st.text_area("Or paste the JD here", height=200, key="jd_text")

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
check_clicked = st.button("🔍 Check Job Compatibility")
st.markdown("</div>", unsafe_allow_html=True)

if check_clicked:
    compatibility_score = "82%"
    suggestions = [
        "Match your title to the job title.",
        "Highlight required skills like Python and SQL.",
        "Emphasize relevant experience in fintech sector."
    ]

    st.markdown("---")
    st.subheader("✅ Results")
    st.markdown(f"**Compatibility Score:** {compatibility_score}")

    st.markdown("**Suggestions for Improvement:**")
    for s in suggestions:
        st.markdown(f"- {s}")

if __name__ == "__main__":
    resume = extractData("./data/resumes/resume_software_engineer_1.pdf")
    jd = extractData("./data/job_descriptions/jd_software_engineer_2.txt")
    TFIDF_score = compareResumeToJd(resume,jd)
    print(f"Similarity Score working: {TFIDF_score}")

    resume_skills = extract_keywords(resume)
    jd_skills = extract_keywords(jd)

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills

    print("✅ Matched:", matched)
    print("❌ Missing:", missing)