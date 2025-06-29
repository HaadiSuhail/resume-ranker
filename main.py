import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 1. Load job description
with open("job_description.txt", "r", encoding="utf-8") as f:
    job_desc = f.read()

# 2. Prepare list of resumes
resume_dir = "resumes"
resume_texts = []
resume_names = []

for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        filepath = os.path.join(resume_dir, filename)
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        resume_texts.append(text)
        resume_names.append(filename)

# 3. Fallback dummy data if no resumes
if not resume_texts:
    print("⚠️ No resumes found, using sample data.")
    resume_texts = [
        "Experienced data scientist with Python skills.",
        "Software engineer with C++ experience.",
        "Machine learning researcher with NLP background."
    ]
    resume_names = ["Resume 1", "Resume 2", "Resume 3"]

# 4. Vectorize
documents = [job_desc] + resume_texts
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# 5. Compute similarity
cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# 6. Create DataFrame
results = pd.DataFrame({
    "Resume": resume_names,
    "Similarity": cos_sim
})

# 7. Sort by similarity
results = results.sort_values(by="Similarity", ascending=False)

# 8. Print results
print(results)

# 9. Save to CSV
results.to_csv("ranking_results.csv", index=False)