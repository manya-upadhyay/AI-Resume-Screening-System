# AI Resume Screening System

## Overview
AI Resume Screening System is a Machine Learning based web application that analyzes resumes and provides intelligent insights. It predicts the resume category, calculates Resume Score and ATS Score, detects skills, recommends missing skills, estimates company match, generates AI suggestions, and provides Gemini AI based resume review.

## Features

- Resume Category Prediction
- Resume Score
- ATS Score
- Skill Detection
- Recommended Skills
- Company Match
- Resume Strength Analysis
- Resume Section Checker
- Gemini AI Resume Review
- PDF Report Download
- Interactive Charts
- Modern Streamlit UI

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- PDFPlumber
- Google Gemini API

## Machine Learning

- TF-IDF Vectorizer
- Label Encoder
- Classification Model

## Project Structure

AI_Resume_Screening/
│
├── app.py
├── utils.py
├── style.css
├── requirements.txt
├── README.md
├── dataset/
├── model/
│ ├── clf.pkl
│ ├── tfidf.pkl
│ └── label_encoder.pkl
└── assets/

## How to Run

pip install -r requirements.txt

streamlit run app.py

## Future Improvements

- Job Recommendation
- Resume Ranking
- LinkedIn Profile Analysis
- Interview Question Generator

## Author

Manya Upadhyay