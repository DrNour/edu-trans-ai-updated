import streamlit as st
import pandas as pd
import difflib
import nltk
import error_categorization as ec  # your module with detect_errors, etc.

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stop_words = set(stopwords.words('english'))

# Fluency scoring function
def fluency_features(text):
    tokens = text.split()
    if not tokens:
        return 0.0

    avg_token_len = sum(len(t) for t in tokens) / len(tokens)
    func_words_count = sum(1 for t in tokens if t.lower() in stop_words)
    func_word_ratio = func_words_count / len(tokens)

    sentences = sent_tokenize(text)
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences),1)

    score = 0.4 * (1 - min(avg_token_len / 20, 1))  # penalize long tokens
    score += 0.4 * min(func_word_ratio / 0.4, 1)    # reward natural function word ratio
    score += 0.2 * (1 - min(abs(avg_sent_len - 15) / 15, 1))  # reward sentence length near 15 tokens

    return round(score, 3)

# Precision and recall for key terms
def precision_recall(student_text, reference_text, key_terms):
    stoks = set(student_text.lower().split())
    rtoks = set(reference_text.lower().split())
    ktoks = set(k.lower() for k in key_terms)

    true_positives = stoks & ktoks
    relevant = rtoks & ktoks

    precision = len(true_positives) / max(len(stoks & ktoks), 1)
    recall = len(true_positives) / max(len(relevant), 1)
    return round(precision, 3), round(recall, 3)

st.title("EduTransAI - Comprehensive Translation Assessment")

uploaded_file = st.file_uploader("Upload your translations CSV file", type=["csv"])

# Example glossary (key terms) - in real app, load from file or database
example_glossary = ["translation", "accuracy", "fluency", "idiom", "syntax", "semantics"]

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            st.error(f"Failed to read the CSV file: {e}")
            st.stop()

    required_cols = ['Student_Translation', 'Reference_Translation']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}")
        st.stop()

    df['Student_Translation'] = df['Student_Translation'].fillna("").astype(str)
    df['Reference_Translation'] = df['Reference_Translation'].fillna("").astype(str)

    # Use your advanced error detection
    df = ec.categorize_errors_dataframe(df,
                                       student_col='Student_Translation',
                                       reference_col='Reference_Translation')

    accuracy_scores = []
    fluency_scores = []
    precisions = []
    recalls = []

    for idx, row in df.iterrows():
        student_text = row['Student_Translation']
        ref_text = row['Reference_Translation']

        # Accuracy score (similarity)
        accuracy = difflib.SequenceMatcher(None, student_text, ref_text).ratio()
        accuracy_scores.append(round(accuracy, 3))

        # Fluency score
        fluency_scores.append(fluency_features(student_text))

        # Precision & recall for glossary terms
        p, r = precision_recall(student_text, ref_text, example_glossary)
        precisions.append(p)
        recalls.append(r)

    df['Accuracy_Score'] = accuracy_scores
    df['Fluency_Score'] = fluency_scores
    df['Glossary_Precision'] = precisions
    df['Glossary_Recall'] = recalls

    st.write("### Full Translation Assessment Results")
    st.dataframe(df[['Student_Translation', 'Reference_Translation', 'Accuracy_Score',
                     'Fluency_Score', 'Glossary_Precision', 'Glossary_Recall', 'Errors_Detected']])

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Results CSV", data=csv,
                       file_name="translation_assessment.csv", mime="text/csv")

else:
    st.info("Please upload a CSV file containing 'Student_Translation' and 'Reference_Translation' columns.")
