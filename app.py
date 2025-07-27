import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import string


class SkillsExtractor:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Comprehensive technical skills database
        self.technical_skills = {
            # Programming Languages
            'python', 'javascript', 'java', 'c#', 'c++', 'c', 'php', 'ruby', 'go', 'rust',
            'swift', 'kotlin', 'typescript', 'r', 'matlab', 'scala', 'perl', 'dart',

            # Frontend Technologies
            'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vue.js', 'vuejs',
            'html', 'html5', 'css', 'css3', 'javascript', 'typescript', 'jquery', 'bootstrap',
            'tailwind', 'tailwindcss', 'sass', 'scss', 'less', 'webpack', 'vite', 'next.js',
            'nextjs', 'nuxt', 'svelte', 'ember', 'backbone', 'ant design', 'material ui',
            'ant-zorro', 'power bi',

            # Backend Technologies
            '.net', '.net core', '.net framework', 'node.js', 'nodejs', 'express', 'fastapi',
            'django', 'flask', 'spring', 'spring boot', 'laravel', 'ruby on rails', 'rails',
            'asp.net', 'nest.js', 'nestjs', 'koa', 'hapi', 'restify',

            # Databases
            'mysql', 'postgresql', 'mongodb', 'sqlite', 'redis', 'oracle', 'sql server',
            'ms sql server', 'mariadb', 'cassandra', 'dynamodb', 'firebase', 'elasticsearch',
            'mongodbcompass', 'sql',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github actions',
            'terraform', 'ansible', 'chef', 'puppet', 'nginx', 'apache', 'lambda', 'ec2',
            'apigateway', 'cognito', 'iam', 'cloudwatch', 'api gateway', 'cloud watch',

            # APIs & Protocols
            'rest api', 'restful', 'graphql', 'soap', 'soap api', 'grpc', 'webhook', 'json',
            'xml', 'yaml', 'microservices',

            # Machine Learning & Data Science
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'opencv', 'scipy', 'jupyter', 'jupyter notebook',
            'machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision',
            'data science', 'artificial intelligence',

            # Testing & Tools
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'jira', 'confluence', 'slack',
            'trello', 'asana', 'postman', 'insomnia', 'swagger', 'jest', 'mocha', 'cypress',
            'selenium', 'junit', 'pytest', 'unittest',

            # Methodologies
            'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd', 'mvp', 'mvc',
            'microservices architecture', 'serverless',

            # Soft Skills
            'leadership', 'team management', 'communication', 'problem solving', 'analytical',
            'time management', 'project management', 'mentoring', 'training', 'collaboration'
        }

        # Create skill variations (e.g., React.js -> react)
        self.skill_variations = {}
        for skill in self.technical_skills:
            variations = [
                skill.lower(),
                skill.replace('.', ''),
                skill.replace(' ', ''),
                skill.replace('-', ''),
                skill.replace('js', ''),
                skill.replace('.js', '')
            ]
            for var in variations:
                if var:
                    self.skill_variations[var] = skill

    def preprocess_text(self, text, light_preprocessing=True):
        """Light preprocessing to preserve technical terms"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        if light_preprocessing:
            # Only remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        else:
            # Heavy preprocessing (use for similarity comparison)
            if self.nlp:
                doc = self.nlp(text)
                tokens = [token.lemma_ for token in doc
                          if not token.is_stop and not token.is_punct
                          and len(token.text) > 2]
                text = ' '.join(tokens)

        return text

    def extract_skills_pattern_based(self, text):
        """Extract skills using pattern matching with comprehensive skill database"""
        text_lower = text.lower()
        found_skills = set()

        # Direct matching
        for skill in self.technical_skills:
            if skill in text_lower:
                found_skills.add(skill)

        # Pattern-based extraction for compound skills
        patterns = [
            r'(\w+\.js)', r'(\w+js)', r'(\w+\.net)', r'(\.net \w+)',
            r'(aws \w+)', r'(\w+ api)', r'(\w+ framework)', r'(\w+ server)',
            r'(\w+ database)', r'(\w+ learning)', r'(\w+ development)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match in self.technical_skills:
                    found_skills.add(match)

        return found_skills

    def extract_skills_context_aware(self, text):
        """Extract skills using context-aware methods"""
        found_skills = set()

        # Look for skills in specific sections
        sections = {
            'skills': r'(?i)skills?\s*:?\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'technologies': r'(?i)technologies?\s*:?\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'experience': r'(?i)experience\s*:?\s*(.*?)(?=\neducation|\nprojects?|$)',
            'requirements': r'(?i)(?:required?\s*)?(?:qualifications?|skills?|requirements?)\s*:?\s*(.*?)(?=\nwhat|\nresponsibilities|\n[A-Z]|$)'
        }

        for section_name, pattern in sections.items():
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                section_skills = self.extract_skills_pattern_based(match)
                found_skills.update(section_skills)

        return found_skills

    def normalize_skills(self, skills):
        """Normalize skill variations to standard forms"""
        normalized = set()
        for skill in skills:
            # Check if it's a known variation
            standard_skill = self.skill_variations.get(skill.lower(), skill)
            normalized.add(standard_skill)
        return normalized

    def extract_all_skills(self, text):
        """Comprehensive skill extraction combining multiple methods"""
        all_skills = set()

        # Method 1: Pattern-based extraction
        pattern_skills = self.extract_skills_pattern_based(text)
        all_skills.update(pattern_skills)

        # Method 2: Context-aware extraction
        context_skills = self.extract_skills_context_aware(text)
        all_skills.update(context_skills)

        # Method 3: N-gram extraction for compound skills
        words = text.lower().split()
        for i in range(len(words)):
            # Check single words
            word = words[i].strip(string.punctuation)
            if word in self.technical_skills:
                all_skills.add(word)

            # Check bigrams and trigrams
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i + 1]}".strip(string.punctuation)
                if bigram in self.technical_skills:
                    all_skills.add(bigram)

            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}".strip(string.punctuation)
                if trigram in self.technical_skills:
                    all_skills.add(trigram)

        return self.normalize_skills(all_skills)

    def calculate_similarity(self, resume_text, jd_text):
        """Calculate similarity with improved preprocessing"""
        # Light preprocessing for skill extraction
        resume_clean = self.preprocess_text(resume_text, light_preprocessing=True)
        jd_clean = self.preprocess_text(jd_text, light_preprocessing=True)

        # Heavy preprocessing for similarity calculation
        resume_processed = self.preprocess_text(resume_text, light_preprocessing=False)
        jd_processed = self.preprocess_text(jd_text, light_preprocessing=False)

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Include bigrams and trigrams
            min_df=1,
            stop_words='english'
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0.0

        return similarity

    def analyze_resume_jd_match(self, resume_text, jd_text):
        """Comprehensive analysis of resume-JD match"""
        # Extract skills from both documents
        resume_skills = self.extract_all_skills(resume_text)
        jd_skills = self.extract_all_skills(jd_text)

        # Find matches and missing skills
        matched_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills - resume_skills
        extra_skills = resume_skills - jd_skills

        # Calculate similarity
        similarity_score = self.calculate_similarity(resume_text, jd_text)

        # Calculate skill match percentage
        if len(jd_skills) > 0:
            skill_match_percentage = len(matched_skills) / len(jd_skills)
        else:
            skill_match_percentage = 0.0

        return {
            'similarity_score': similarity_score,
            'skill_match_percentage': skill_match_percentage,
            'resume_skills': sorted(resume_skills),
            'jd_skills': sorted(jd_skills),
            'matched_skills': sorted(matched_skills),
            'missing_skills': sorted(missing_skills),
            'extra_skills': sorted(extra_skills),
            'total_resume_skills': len(resume_skills),
            'total_jd_skills': len(jd_skills),
            'total_matched_skills': len(matched_skills)
        }


# Example usage
def main():
    # Your resume and JD text
    resume_text = """
    Sarib Bin Nasir
    Software Engineer

    Summary
    Results‑driven Software Engineer with over 3 years of experience in designing, developing, and maintaining full‑stack and enterprise‑level applications...

    Skills
    Back‑end .NET Core, .NET Framework, C#, Fullstack Web Development, Python, Fast API, REST API, SOAP API, Node.js, AWS (Lambda, ApiGateway, Cognito, IAM, Cloud Watch)
    Front‑end Angular, React, Next.JS, Web (HTML, CSS, JS, TS), JavaScript, Ant‑Zorro, Tailwind CSS, Power BI
    Database MS SQL Server, SQL, MongoDB, MongoDBCompass, MYSQL
    Libraries Jupyter Notebook, OpenCV, Sci‑kit learn, NumPy, Pandas, SciPy, Matplotlib, TensorFlow, Keras, Pytorch, sklearn
    """

    jd_text = """
    Front-End Developer (Experience with React or Angular is a plus)

    Required Qualifications:
    Basic understanding of HTML5, CSS3, and JavaScript.
    Familiarity with at least one modern front-end framework like React, Angular, or Vue.js.
    Good communication and problem-solving skills.
    Exposure to REST APIs and JSON.
    Understanding of version control (Git/GitHub).
    """

    extractor = SkillsExtractor()
    results = extractor.analyze_resume_jd_match(resume_text, jd_text)

    print(f"Similarity Score: {results['similarity_score']:.3f}")
    print(f"Skill Match Percentage: {results['skill_match_percentage']:.1%}")
    print(f"\n✅ Matched Skills ({len(results['matched_skills'])}): {results['matched_skills']}")
    print(f"\n❌ Missing Skills ({len(results['missing_skills'])}): {results['missing_skills']}")
    print(
        f"\n➕ Extra Skills in Resume ({len(results['extra_skills'])}): {results['extra_skills'][:10]}...")  # Show first 10


if __name__ == "__main__":
    main()