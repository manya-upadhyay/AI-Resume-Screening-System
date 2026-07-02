from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

import streamlit as st
import joblib
import pdfplumber
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#api key
def gemini_resume_review(resume_text):
    prompt = f"""
    You are an expert ATS Resume Reviewer.

    Analyze this resume and provide:
    1. Resume Summary
    2. Strengths
    3. Weaknesses
    4. Missing Skills
    5. ATS Improvement Tips
    6. Interview Preparation Tips

    Resume:
    {resume_text}
    """

    try:
        response = model_gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 700
            }
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini AI is temporarily unavailable.\n\nError: {e}"

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/clf.pkl")
tfidf = joblib.load("model/tfidf.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

st.title("📄 AI Resume Screening System")
st.write("Upload your resume in PDF format and let AI analyze it.")

with st.sidebar:
    st.title("📄 AI Resume Screening")
    st.markdown("---")
    st.info("Upload your resume to get AI-powered analysis.")
    st.markdown("### Features")
    st.write("✅ Resume Category")
    st.write("✅ Resume Score")
    st.write("✅ ATS Score")
    st.write("✅ Company Match")
    st.write("✅ AI Suggestions")

# -----------------------------
# Skills List
# -----------------------------
skills = [
    "python", "java", "c++", "sql", "html", "css",
    "javascript", "react", "machine learning",
    "deep learning", "data analysis", "pandas",
    "numpy", "scikit-learn", "communication",
    "git", "excel", "power bi"
]

company_requirements = {
    "Google": ["python", "sql", "machine learning", "git", "communication"],
    "Microsoft": ["python", "c++", "sql", "git", "excel"],
    "Amazon": ["java", "sql", "data analysis", "communication"],
    "TCS": ["java", "html", "css", "sql"],
    "Infosys": ["python", "java", "communication", "excel"]
}

#ATS score
def calculate_ats_score(resume, found_skills):
    score = 0

    # Resume Length
    words = len(resume.split())

    if words >= 300:
        score += 20
    elif words >= 200:
        score += 15
    else:
        score += 10

    # Skills
    score += min(len(found_skills) * 5, 40)

    # Important Sections
    sections = [
        "education",
        "experience",
        "project",
        "skills",
        "certification"
    ]

    for section in sections:
        if section.lower() in resume.lower():
            score += 8

    return min(score, 100)

#resume suggestions
def resume_suggestions(found_skills, ats_score):
    suggestions = []

    if "python" not in found_skills:
        suggestions.append("Learn Python and add Python projects.")

    if "sql" not in found_skills:
        suggestions.append("Add SQL to improve your database skills.")

    if "git" not in found_skills:
        suggestions.append("Mention Git/GitHub in your resume.")

    if "communication" not in found_skills:
        suggestions.append("Highlight your communication skills.")

    if ats_score < 70:
        suggestions.append("Your ATS score is low. Add more relevant skills, projects and certifications.")

    if ats_score >= 90:
        suggestions.append("Excellent resume! You are close to being interview-ready.")

    return suggestions

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# -----------------------------
# Skill Detection
# -----------------------------
def detect_skills(text):
    text = text.lower()
    found_skills = []

    for skill in skills:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills

def company_match(found_skills):
    scores = {}

    for company, required_skills in company_requirements.items():
        matched = len(set(found_skills) & set(required_skills))
        score = int((matched / len(required_skills)) * 100)
        scores[company] = score

    return scores

def check_resume_sections(resume):
    sections = [
        "summary",
        "education",
        "experience",
        "projects",
        "skills",
        "certifications"
    ]

    missing = []

    text = resume.lower()

    for section in sections:
        if section not in text:
            missing.append(section)

    return missing

#generate report
def generate_report(category, resume_score, ats_score, found_skills, recommended_skills):
    doc = SimpleDocTemplate("Resume_Analysis_Report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AI Resume Screening Report</b>", styles["Title"]))
    story.append(Paragraph(f"<b>Predicted Category:</b> {category}", styles["Normal"]))
    story.append(Paragraph(f"<b>Resume Score:</b> {resume_score}/100", styles["Normal"]))
    story.append(Paragraph(f"<b>ATS Score:</b> {ats_score}/100", styles["Normal"]))

    story.append(Paragraph("<br/><b>Detected Skills:</b>", styles["Heading2"]))
    story.append(Paragraph(", ".join(found_skills), styles["Normal"]))

    story.append(Paragraph("<br/><b>Recommended Skills:</b>", styles["Heading2"]))
    story.append(Paragraph(", ".join(recommended_skills), styles["Normal"]))

    doc.build(story)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Resume"):

    if uploaded_file is not None:

        resume = extract_text_from_pdf(uploaded_file)

        resume_vector = tfidf.transform([resume])

        prediction = model.predict(resume_vector)

        category = label_encoder.inverse_transform(prediction)

        st.success(f"🎯 Predicted Category: {category[0]}")

        # -----------------------------
        # Resume Score
        # -----------------------------
        found_skills = detect_skills(resume)

        score = min(len(found_skills) * 10, 100)

        st.subheader("⭐ Resume Score")
        st.progress(score / 100)
        st.write(f"Your Resume Score: **{score}/100**")

        # -----------------------------
        # Skills Found
        # -----------------------------
        st.subheader("💻 Detected Skills")

        if found_skills:
            for skill in found_skills:
                st.success(skill)
        else:
            st.error("No important skills detected.")

        # -----------------------------
        # Missing Skills
        # -----------------------------

        st.subheader("📌 Recommended Skills")
        missing = [skill for skill in skills if skill not in found_skills]

        if missing:
            for skill in missing[:5]:
                st.warning(skill)
        else:
            st.success("Excellent! No important skills missing.")

        st.subheader("🥧 Skills Overview")

        labels = ["Detected", "Recommended"]
        sizes = [len(found_skills), len(missing[:5])]

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax2.axis("equal")

        st.pyplot(fig2)

        #company match

        st.subheader("🏢 Company Match")

        scores = company_match(found_skills)

        for company, score in scores.items():
            st.write(f"**{company}** : {score}%")
            st.progress(score / 100)

        st.subheader("📊 Company Match Chart")

        companies = list(scores.keys())
        values = list(scores.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(companies, values)

        ax.set_ylabel("Match %")
        ax.set_xlabel("Companies")
        ax.set_title("Company Match Score")

        st.pyplot(fig)


         #ats score
        st.subheader("📊 ATS Score")

        ats_score = calculate_ats_score(resume, found_skills)

        st.progress(ats_score / 100)

        st.success(f"ATS Score: {ats_score}/100")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Resume Score", f"{score}/100")

        with col2:
            st.metric("ATS Score", f"{ats_score}/100")

        with col3:
            st.metric("Skills Found", len(found_skills))

         #suggestions
        st.subheader("🤖 AI Resume Suggestions")

        suggestions = resume_suggestions(found_skills, ats_score)

        for suggestion in suggestions:
            st.info(suggestion)

        #missing sections in resume
        missing_sections = check_resume_sections(resume)

        st.subheader("📋 Resume Sections Check")

        if len(missing_sections) == 0:
            st.success("✅ All important sections are present.")
        else:
            for section in missing_sections:
                st.warning(f"Missing: {section.title()}")

        #strength
        st.subheader("📈 Resume Strength")

        if ats_score >= 90:
            st.success("🟢 Excellent Resume")
        elif ats_score >= 75:
            st.warning("🟡 Good Resume - Minor improvements needed")
        else:
            st.error("🔴 Resume Needs Improvement")

        #report
        generate_report(
            category[0],
            score,
            ats_score,
            found_skills,
            missing
        )

        with open("Resume_Analysis_Report.pdf", "rb") as pdf_file:
            st.download_button(
                label="📥 Download Analysis Report",
                data=pdf_file,
                file_name="Resume_Analysis_Report.pdf",
                mime="application/pdf"
            )

        #api key
        st.subheader("🤖 Gemini AI Resume Review")

        with st.spinner("Generating AI Review..."):
            review = gemini_resume_review(resume)

        st.markdown(review)

    else:
        st.warning("Please upload a PDF resume.")

