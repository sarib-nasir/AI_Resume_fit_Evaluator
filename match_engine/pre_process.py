import os.path
import re
import spacy
import string
import fitz

nlp = spacy.load("en_core_web_sm")

def extractData(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text_data =  extractDataFromFile(path)
    elif ext == ".txt":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text_data =  f.read()

    data = preProcess(text_data,path)
    return data

def extractDataFromFile(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def preProcess(text,doc_path):
    # text = open(file_path,'r')
    # text = text.read()
    text = text.lower()

    re.sub(r'\S+@\S+', '', text)
    re.sub(r'http\S+', '', text)
    re.sub(r'\b\d{10,}\b', '', text)

    text = text.translate(str.maketrans('','',string.punctuation))


    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    print(doc_path)
    print(tokens)
    return ' '.join(tokens)

#
# if __name__ == "__main__":
#     resume = "data/resumes/resume_software_engineer_1.txt"
#     jd = "data/job_descriptions/jd_software_1.txt"
#     score = preProcess(resume)
#     print(f"Similarity Score: {score}")