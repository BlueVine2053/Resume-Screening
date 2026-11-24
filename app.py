"""Optional Streamlit demo. Scores are similarities, not hiring recommendations."""
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
from src.data import load_dataset, prepare
from src.matching import Matcher

st.set_page_config(page_title='Resume–Job Matching Demo', layout='centered')
st.title('Resume Screening and Job Matching')
st.caption('V1 keyword matching vs. V2 TF-IDF cosine similarity.')

# CR-002: Persistent disclaimer and keyword matching limitation.
st.info(
    'These are similarity scores, not verified fit classifications '
    'or hiring recommendations.'
)
st.caption(
    'V1 uses exact-word matching and does not recognize synonyms. '
    'A low similarity score does not prove that a candidate lacks a skill.'
)

@st.cache_resource(show_spinner=False)
def get_matcher():
    artifact = Path('results/final/model.joblib')
    if artifact.exists():
        return joblib.load(artifact)['matcher']
    # Fallback for first-time demo before running the final experiment.
    jobs = ['python sql data analysis', 'java spring backend development',
            'patient care nursing clinical records', 'project management communication',
            'cybersecurity network monitoring', 'finance accounting reporting']
    resumes = ['python sql analyst', 'java software engineer', 'registered nurse patient care',
               'project manager', 'security network analyst', 'financial reporting specialist']
    return Matcher(jobs, resumes)

model = get_matcher()

if Path('results/final/model.joblib').exists():
    st.success('Using the trained evaluation matcher (same scoring functions).')
else:
    st.warning('Using synthetic demonstration matcher. Run the final experiment to load the trained model.')

mode = st.radio('Input method', ['Enter text', 'Select sample pair'], horizontal=True)

if mode == 'Select sample pair':
    df, _ = prepare(load_dataset(path='data/sample_pairs.csv'))
    selected = st.selectbox('Choose a sample pair', range(len(df)), format_func=lambda i: f"Pair {i+1} — {df.iloc[i]['label']}")
    resume = df.iloc[selected]['resume']; job = df.iloc[selected]['job']
    st.text_area('Resume', resume, height=120, disabled=True)
    st.text_area('Job description', job, height=120, disabled=True)
else:
    resume = st.text_area('Resume text', height=140)
    job = st.text_area('Job description', height=140)

if st.button('Compare matching scores'):
    try:
        scores = model.score(resume, job)
        st.metric('V1 keyword overlap', 'Not scoreable' if scores['v1'] is None else f"{scores['v1']:.3f}")
        st.metric('V2 TF-IDF cosine similarity', f"{scores['v2']:.3f}")
    except ValueError as error:
        st.error(str(error))
