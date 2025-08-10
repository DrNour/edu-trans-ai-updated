from difflib import SequenceMatcher
import re

# --- Arabic-friendly tokenization ---
def tokenize_arabic(text):
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

# --- Error category detection ---
def categorize_errors(src_text, student_text):
    errors = []
    src_tokens = src_text.lower().split()
    stu_tokens = tokenize_arabic(student_text)

    # 1. Omission
    if len(stu_tokens) < len(src_tokens) * 0.9:
        errors.append("Omission")
    # 2. Addition
    if len(stu_tokens) > len(src_tokens) * 1.1:
        errors.append("Addition")
    # 3. Idioms
    idioms = ["hit the nail on the head", "once in a blue moon", "spill the beans"]
    for idiom in idioms:
        if idiom in src_text.lower() and idiom not in " ".join(stu_tokens).lower():
            errors.append("Idiom Loss")
            break
    # 4. False friends
    false_friends = {"actual": "حقيقي", "eventually": "أخيراً", "sensible": "معقول"}
    for eng, ar in false_friends.items():
        if eng in src_text.lower() and ar not in student_text:
            errors.append("False Friend")
    # 5. Low lexical similarity
    common_tokens = set(src_tokens) & set(stu_tokens)
    lexical_overlap = len(common_tokens) / max(len(src_tokens), 1)
    if lexical_overlap < 0.3:
        errors.append("Low Lexical Similarity")

    return errors

# --- Main app ---
import streamlit as st
import pandas as pd

st.title("EduTransAI - Translation Assessment Tool")

uploaded_file = st.file_uploader("Upload your translations CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')

    st.success("CSV file loaded successfully! Here's a preview:")
    st.dataframe(df)

    required_columns = ['Student_Translation', 'Reference_Translation']
    if all(col in df.columns for col in required_columns):
        df['Similarity'] = 0.0
        df['Errors Detected'] = ""

        for i, row in df.iterrows():
            s = str(row['Student_Translation'])
            r = str(row['Reference_Translation'])
            df.at[i, 'Similarity'] = SequenceMatcher(None, s, r).ratio()
            errors = categorize_errors(r, s)
            df.at[i, 'Errors Detected'] = ", ".join(errors) if errors else "No major errors"

        st.write("### Assessment Results with Error Categories")
        st.dataframe(df[['Student_Translation', 'Reference_Translation', 'Similarity', 'Errors Detected']])

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='translation_assessment_results.csv',
            mime='text/csv'
        )
    else:
        st.error(f"CSV file must contain: {required_columns}")
else:
    st.info("Please upload a CSV file containing 'Student_Translation' and 'Reference_Translation' columns.")
