# Intelligent FAQ Generation System Using Natural Language Processing (NLP)

A complete, lightweight, and offline B.Tech Natural Language Processing (NLP) microproject that automatically extracts key concepts and generates high-quality Frequently Asked Questions (FAQs) and answers directly from uploaded documents.

---

## 1. Project Overview & Abstract

In academia and corporate environments, vast amounts of unstructured textual information (syllabi, technical reports, institutional guidelines, research summaries) require significant human effort to study and summarize. 

The **Intelligent FAQ Generation System** addresses this problem by automating the formulation of interrogative question-and-answer pairs directly from source documents. Unlike generic question answering systems that require heavy cloud APIs, deep learning models, or external keys, this system utilizes fundamental, transparent, and explainable **Natural Language Processing (NLP)** techniques:
- **PyMuPDF stream extraction** for document ingest (PDF and TXT).
- **NLTK sentence & word tokenization** with stop-word filtration.
- **Scikit-Learn TF-IDF vectorization** for domain keyword extraction.
- **Syntactic and rule-based linguistic transformation patterns** to convert factual statements into grammatical questions (What, Who, Where, When, How many, Why, How).
- **Extractive answering** that ensures zero hallucination by strictly drawing answers directly from supporting source sentences.

---

## 2. Problem Statement & Objectives

### Problem Statement
Manual creation of FAQs from academic textbooks, research papers, or organizational documents is repetitive, time-consuming, and prone to subjective omissions.

### Objectives
1. **Automated Parsing**: Extract text robustly from TXT and PDF documents while handling corrupted, scanned, or empty documents gracefully.
2. **Text Normalization**: Clean whitespace, repair hyphenated line breaks, and tokenize into sentences and words.
3. **Keyword Identification**: Compute Term Frequency-Inverse Document Frequency (TF-IDF) to find dominant domain concepts.
4. **Question Generation**: Apply rule-based syntactic transformations to produce coherent questions across key interrogative types (*Where, When, Who, What, How many, Why*).
5. **Faithful Answering**: Pair every question with its exact source sentence to guarantee 100% factual accuracy without hallucination.
6. **Student-Friendly UI & Export**: Provide a clean Streamlit interface with metrics, keyword inspection, and CSV export.

---

## 3. Technology Stack

- **Programming Language:** Python 3.10+ (tested on Python 3.13)
- **User Interface:** Streamlit
- **NLP Library:** Natural Language Toolkit (NLTK)
- **Feature Extraction & Machine Learning:** Scikit-Learn (`TfidfVectorizer`)
- **PDF Extraction:** PyMuPDF (`pymupdf`)
- **Data Export & Handling:** Pandas

> **Note:** Requires **no paid APIs, API keys, external cloud accounts, or large language models**. Runs completely locally and deterministically.

---

## 4. NLP Techniques Implemented

### A. Document Normalization & Cleaning
- Uniform line-ending conversion (`\r\n` -> `\n`).
- Hyphenated word repair across line wraps (e.g., `com-\nmunication` -> `communication`).
- Whitespace collapsing and control character filtering.

### B. Sentence & Word Tokenization
- Uses NLTK's `sent_tokenize` (Punkt tokenizer) to split text into distinct sentence units.
- Filters out sub-sentence fragments, bullet headers, or isolated numbers (< 5 words).
- Uses `word_tokenize` for lexical decomposition and character analysis.

### C. Stop-Word Removal & TF-IDF Keyword Extraction
- Eliminates non-informative grammatical noise using NLTK English stop words augmented with custom conversational stops (`also`, `many`, `often`, `used`).
- Computes **Term Frequency-Inverse Document Frequency (TF-IDF)** using `scikit-learn`:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- Identifies unigrams and bigrams with highest weights to isolate core subjects.

### D. Rule-Based Interrogative Question Generation
The system matches declarative grammatical patterns against deterministic question templates:

| Question Type | Pattern / Trigger | Example Source Sentence | Generated Question |
| :--- | :--- | :--- | :--- |
| **Where** | `[Subject] is/was located in [Location]` | *The college is located in Ashta, Maharashtra.* | **Where is college located?** |
| **When** | `[Subject] was founded/established in [Year]` | *The university was established in 1983.* | **When was university established?** |
| **Who** | `[Person] was a/an [Profession]` | *Dr. APJ Abdul Kalam was an Indian aerospace scientist...* | **Who was Dr. APJ Abdul Kalam?** |
| **How many** | `[Subject] has/contains [Count] [Items]` | *The engineering library contains over 50,000 books...* | **How many books does engineering library contain?** |
| **What (Def)** | `[Subject] is/are a/an [Concept]` | *Natural Language Processing (NLP) is a branch of AI...* | **What is Natural Language Processing (NLP)?** |
| **What (Obj)** | `The objective/goal of [Subject] is to [Verb]` | *The primary objective of NLP is to enable machines...* | **What is the objective of NLP?** |
| **What (Cap)** | `[Subject] offers/provides [Feature]` | *The institution offers undergraduate engineering programs.* | **What does institution offer?** |
| **Why** | `[Subject] is important because [Clause]` | *Feature engineering is important because high quality...* | **Why is Feature engineering considered important?** |

---

## 5. Project Directory Structure

```text
NLP/
├── app.py                 # Streamlit web application & user interface
├── document_loader.py     # PDF & TXT extraction, encoding fallbacks, validation
├── nlp_processor.py       # Sentence tokenization, stop-words, TF-IDF calculation
├── faq_generator.py       # Rule-based syntactic question engines & answer pairing
├── sample_document.txt    # Provided academic & NLP demonstration text
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignored files & environments
├── README.md              # Microproject documentation & viva guide
├── create_test_pdf.py     # Script to generate test PDF
└── sample_test.pdf        # Generated sample PDF for verification
```

---

## 6. Installation & Execution Instructions

### Step 1: Open Terminal in the Project Directory
```powershell
cd c:\antigravity\NLP
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Web Application
```powershell
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## 7. Sample Demonstration (Input vs. Output)

### Sample Input Text (`sample_document.txt`):
> "The college is located in Ashta, Maharashtra. The institution offers undergraduate engineering programs. The university was established in 1983. Dr. APJ Abdul Kalam was an Indian aerospace scientist who inspired millions of students. The engineering library contains over 50,000 books and reference journals. Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand human language."

### Generated Output:
1. **Q:** Where is college located?  
   **A:** The college is located in Ashta, Maharashtra.  
   *(Rule: Location (Where))*
2. **Q:** What does institution offer?  
   **A:** The institution offers undergraduate engineering programs.  
   *(Rule: Functional (Service/Offer))*
3. **Q:** When was university established?  
   **A:** The university was established in 1983.  
   *(Rule: Temporal (When))*
4. **Q:** Who was Dr. APJ Abdul Kalam?  
   **A:** Dr. APJ Abdul Kalam was an Indian aerospace scientist who inspired millions of students.  
   *(Rule: Person (Who))*
5. **Q:** How many books does engineering library contain?  
   **A:** The engineering library contains over 50,000 books and reference journals.  
   *(Rule: Quantity (How many))*
6. **Q:** What is Natural Language Processing (NLP)?  
   **A:** Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language.  
   *(Rule: Definitional (Is/Are))*

---

## 8. Viva Questions & Explanations

1. **What is TF-IDF and why is it used here?**  
   *TF-IDF assesses term importance by multiplying term frequency with inverse document frequency. In this project, it isolates key subject terminology from background noise without manual hardcoding.*
2. **Why use rule-based NLP instead of an LLM?**  
   *Rule-based systems are deterministic, completely transparent, explainable for viva evaluation, zero-cost, require no GPUs/APIs, and produce 0% hallucinations because answers are directly extractive.*
3. **How does the system ensure answers are not fabricated?**  
   *Every answer is extracted directly from the original source sentence matched by the rule engine. The application links each generated question to its exact source sentence.*

---

## 9. Applications, Limitations & Future Scope

### Applications
- Automatic revision quiz and FAQ generation for college lectures and textbooks.
- Organization policy summaries and student handbook search assistance.
- Rapid review of research papers and technical reports.

### Limitations
- Requires structured, grammatical sentences in English (colloquial shorthand may yield fewer pattern matches).
- Cannot extract text from pure image-scanned PDFs without external OCR engines like Tesseract.

### Future Scope
- Integration of Tesseract OCR for scanned physical document processing.
- Dependency parsing using SpaCy for deeper clause subordination.
- Multi-lingual question generation for regional languages.
