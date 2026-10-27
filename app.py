import streamlit as st
import pandas as pd
import os
from document_loader import (
    extract_text_from_pdf,
    extract_text_from_txt,
    clean_text,
    validate_text
)
from nlp_processor import tokenize_sentences, tokenize_words
from faq_generator import generate_faqs

# Configure Page
st.set_page_config(
    page_title="Intelligent FAQ Generation System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design & Typography
st.markdown("""
<style>
    /* Main Layout Aesthetics */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero-title {
        font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.6rem;
    }
    .stat-badge {
        display: inline-block;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 8px;
        margin-right: 8px;
    }
    .faq-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease-in-out;
    }
    .faq-box:hover {
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.07);
    }
    .faq-pill {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .faq-q {
        font-size: 1.08rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .faq-a {
        font-size: 0.96rem;
        color: #334155;
        line-height: 1.55;
    }
    .kw-tag {
        display: inline-block;
        background: linear-gradient(135deg, #F3E8FF 0%, #E9D5FF 100%);
        color: #6B21A8;
        border: 1px solid #D8B4FE;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ FAQ Generation Controls")
    
    faq_limit = st.slider(
        "Max Number of FAQs:",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
        help="Select the maximum number of FAQ pairs to generate from the document."
    )
    
    st.divider()
    st.markdown("### 🧪 Quick Demonstration")
    use_sample = st.checkbox("Load Sample Document (`sample_document.txt`)", value=False)
    
    st.divider()
    st.markdown("### 🎓 Academic Evaluation Details")
    st.info("""
    **Project Title:**  
    Intelligent FAQ Generation System Using NLP
    
    **Core Techniques:**
    - Tokenization (Sentence & Word)
    - Stop-Word Filtering
    - Scikit-Learn TF-IDF Keyword Extraction
    - Rule-based Syntactic Question Formulation
    - Extractive Sentence Sourcing
    - PyMuPDF Text Stream Parsing
    """)
    st.caption("B.Tech NLP Microproject | Fully Offline & Deterministic")

# Main Title Section
st.markdown('<div class="hero-title">🧠 Intelligent FAQ Generation System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Automatic question and extractive answer generation using rule-based NLP and TF-IDF</div>', unsafe_allow_html=True)

# Layout: Upload & Input
col_left, col_right = st.columns([1.1, 0.9], gap="large")

file_name = None
raw_content = ""

with col_left:
    st.markdown("### 📄 1. Document Upload")
    
    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_document.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
            file_name = "sample_document.txt"
            st.success("Loaded **sample_document.txt** (College & NLP Overview) successfully!")
        else:
            st.error("Sample document file not found.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a document (PDF or TXT)",
            type=["pdf", "txt"],
            help="Upload a syllabus, lecture notes, college article, or technical document."
        )
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    raw_content = extract_text_from_pdf(file_bytes)
                else:
                    raw_content = extract_text_from_txt(file_bytes)
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

with col_right:
    st.markdown("### 📊 2. Document Metrics & Inspection")
    if raw_content:
        cleaned_text = clean_text(raw_content)
        is_valid, validation_msg = validate_text(cleaned_text)
        
        if not is_valid:
            st.warning(f"⚠️ {validation_msg}")
        else:
            sentences = tokenize_sentences(cleaned_text)
            words = tokenize_words(cleaned_text)
            char_count = len(cleaned_text)
            
            st.markdown(f"""
            <div>
                <span class="stat-badge">📄 File: <b>{file_name}</b></span>
                <span class="stat-badge">📝 Sentences: <b>{len(sentences)}</b></span>
                <span class="stat-badge">🔤 Words: <b>{len(words)}</b></span>
                <span class="stat-badge">🔡 Characters: <b>{char_count}</b></span>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            with st.expander("👁️ Preview Cleaned Document Text", expanded=False):
                st.text_area("Extracted Text", cleaned_text, height=220, disabled=True)
    else:
        st.info("💡 Upload a PDF or TXT file, or check **'Load Sample Document'** to test the system.")

st.divider()

# Generation Action
if raw_content:
    cleaned_text = clean_text(raw_content)
    is_valid, validation_msg = validate_text(cleaned_text)
    
    if is_valid:
        if st.button("⚡ Generate FAQs From Document", type="primary", use_container_width=True):
            with st.spinner("Tokenizing sentences, extracting TF-IDF keywords, and applying syntactic question rules..."):
                results = generate_faqs(cleaned_text, max_faqs=faq_limit)
            
            if not results["success"]:
                st.error(f"⚠️ {results['message']}")
            else:
                faqs = results["faqs"]
                keywords = results["keywords"]
                
                # Display Keywords
                st.markdown("### 🔑 Extracted Domain Keywords (TF-IDF)")
                st.markdown("Salient terms automatically identified across document sentences using Term Frequency-Inverse Document Frequency:")
                
                kw_badges = " ".join([f'<span class="kw-tag">{kw} <small>({score:.2f})</small></span>' for kw, score in keywords])
                st.markdown(kw_badges, unsafe_allow_html=True)
                st.write("")
                
                # Display Generated FAQs
                st.markdown(f"### 📋 Generated FAQs ({len(faqs)} Pairs Generated)")
                
                for idx, item in enumerate(faqs, 1):
                    st.markdown(f"""
                    <div class="faq-box">
                        <span class="faq-pill">{item['rule']}</span>
                        <div class="faq-q">Q{idx}. {item['question']}</div>
                        <div class="faq-a"><b>Answer:</b> {item['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # CSV Export Option
                df = pd.DataFrame(faqs)
                df_export = df.rename(columns={
                    "question": "Question",
                    "answer": "Answer",
                    "rule": "Linguistic Rule Applied"
                })
                csv_bytes = df_export.to_csv(index=False).encode('utf-8')
                
                clean_filename = os.path.splitext(file_name)[0] if file_name else "document"
                st.download_button(
                    label="📥 Download Generated FAQs as CSV",
                    data=csv_bytes,
                    file_name=f"{clean_filename}_faqs.csv",
                    mime="text/csv",
                    use_container_width=True
                )
