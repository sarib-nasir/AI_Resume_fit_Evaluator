import yake

def extract_keywords(text, max_keywords=20):
    kw_extractor = yake.KeywordExtractor(n=1, top=max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return set([kw.lower() for kw, score in keywords])