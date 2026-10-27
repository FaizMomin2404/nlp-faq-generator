import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Set

# Ensure standard NLTK resources are available
def init_nltk():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

init_nltk()

def get_sentences(text: str) -> List[str]:
    """
    Splits text into meaningful sentences, filtering out very short fragments or headers.
    """
    raw_sentences = sent_tokenize(text)
    clean_sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Keep sentences that have at least 5 words and finish with alphanumeric or punctuation
        if len(s_clean.split()) >= 5 and any(c.isalpha() for c in s_clean):
            # Clean internal excessive whitespace
            s_clean = re.sub(r"\s+", " ", s_clean)
            clean_sentences.append(s_clean)
    return clean_sentences

def extract_top_keywords(sentences: List[str], top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Extracts top keywords using TF-IDF across document sentences.
    """
    if not sentences:
        return []
    
    stop_words_list = list(stopwords.words('english'))
    # Include punctuation and common non-informative words
    custom_stops = {'use', 'using', 'used', 'also', 'many', 'much', 'like', 'such', 'one', 'two', 'often', 'can', 'may'}
    all_stops = list(set(stop_words_list).union(custom_stops))
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=all_stops,
            ngram_range=(1, 2),  # unigrams and bigrams
            max_df=0.85,
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_n]
    except Exception:
        # Fallback if vocabulary is empty or too small
        return []

def _clean_subject(text: str) -> str:
    """
    Clean captured subject phrase: remove leading punctuation, conjunctions, and trim.
    """
    s = text.strip()
    # Strip leading/trailing non-alphanumeric except hyphens
    s = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", s)
    # Strip leading determiners if any
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()

def generate_question_from_sentence(sentence: str, keywords: Set[str]) -> Tuple[str, str]:
    """
    Applies rule-based linguistic patterns to generate a relevant question.
    Returns (question, rule_name). If no pattern matches, returns empty string.
    """
    s = sentence.strip()
    if s.endswith('.'):
        s_no_punct = s[:-1]
    else:
        s_no_punct = s

    # Rule 1: Definitional patterns -> "X is/are defined as Y" or "X refers to Y"
    # Matches: "Natural Language Processing (NLP) is a branch of artificial intelligence..."
    m_refers = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,45}?)\s+(?:refers to|means)\s+(.+)", s_no_punct, re.IGNORECASE)
    if m_refers:
        subject = _clean_subject(m_refers.group(1))
        if len(subject.split()) <= 6:
            return f"What does {subject} refer to?", "Definitional (Refers to)"

    m_is_a = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,45}?)\s+(is|are)\s+(a|an|the|defined as|known as)\s+(.+)", s_no_punct, re.IGNORECASE)
    if m_is_a:
        subject = _clean_subject(m_is_a.group(1))
        verb = m_is_a.group(2).lower()
        if len(subject.split()) <= 6:
            return f"What {verb} {subject}?", "Definitional (Is/Are)"

    # Rule 2: Purpose and Objective patterns -> "The primary objective of X is to Y"
    m_purpose = re.match(r"^(?:The\s+)?(primary\s+|main\s+)?(objective|goal|purpose|aim)\s+of\s+([a-zA-Z0-9\s\(\)\-]+?)\s+is\s+to\s+(.+)", s_no_punct, re.IGNORECASE)
    if m_purpose:
        subject = _clean_subject(m_purpose.group(3))
        term = m_purpose.group(2).lower()
        if len(subject.split()) <= 6:
            return f"What is the main {term} of {subject}?", "Purpose/Objective"

    # Rule 3: Functionality / Capability patterns -> "X enables / helps / allows Y"
    m_func = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+(enables|helps|allows|assists|empowers)\s+(.+)", s_no_punct, re.IGNORECASE)
    if m_func:
        subject = _clean_subject(m_func.group(1))
        if len(subject.split()) <= 6:
            return f"How does {subject} work or help?", "Functional/Capability"

    # Rule 4: Importance / Reason -> "X is important because..." or "X is used for..."
    m_because = re.search(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+is\s+(important|essential|crucial|needed)\s+because\s+(.+)", s_no_punct, re.IGNORECASE)
    if m_because:
        subject = _clean_subject(m_because.group(1))
        if len(subject.split()) <= 6:
            return f"Why is {subject} considered {m_because.group(2).lower()}?", "Causal/Reason"

    m_used_for = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+is\s+(?:widely\s+)?used\s+(?:to|for)\s+(.+)", s_no_punct, re.IGNORECASE)
    if m_used_for:
        subject = _clean_subject(m_used_for.group(1))
        if len(subject.split()) <= 6:
            return f"What is {subject} used for?", "Usage"

    # Rule 5: Fallback Keyword & POS-based question generation
    # Extract candidate noun phrase from the beginning or based on top TF-IDF keywords
    try:
        tokens = word_tokenize(s)
        tagged = nltk.pos_tag(tokens)
        
        # Check if first 1-3 words form a noun phrase (e.g. NNP, NN, JJ NN)
        candidate_tokens = []
        for word, tag in tagged[:4]:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']:
                candidate_tokens.append(word)
            elif candidate_tokens and tag in ['VBZ', 'VBP', 'VBD', 'IN']:
                break
        
        if candidate_tokens and len(candidate_tokens) <= 4:
            cand = _clean_subject(" ".join(candidate_tokens))
            cand_lower = cand.lower()
            # If candidate matches any of our extracted keywords or is prominent
            for kw in keywords:
                if kw in cand_lower or cand_lower in kw:
                    return f"What does the document state regarding {cand}?", "Keyword/Concept"
    except Exception:
        pass

    # Generic informative fallback if sentence is strong
    return "", ""

def generate_faqs(text: str, max_faqs: int = 10) -> Dict:
    """
    Main pipeline:
    1. Preprocesses text and splits into sentences.
    2. Computes TF-IDF keywords.
    3. Iterates through sentences and generates rule-based questions.
    4. Pairs each question with its exact supporting sentence answer.
    5. Returns structured results for the Streamlit UI.
    """
    sentences = get_sentences(text)
    if not sentences:
        return {
            "success": False,
            "message": "No valid sentences could be extracted from the document.",
            "faqs": [],
            "keywords": [],
            "total_sentences": 0
        }
    
    keyword_scores = extract_top_keywords(sentences, top_n=20)
    keywords_set = {k[0].lower() for k in keyword_scores}

    faqs = []
    seen_questions = set()
    
    for sentence in sentences:
        question, rule_name = generate_question_from_sentence(sentence, keywords_set)
        
        if question and question not in seen_questions:
            seen_questions.add(question)
            faqs.append({
                "question": question,
                "answer": sentence,
                "rule": rule_name
            })
            if len(faqs) >= max_faqs:
                break

    # If rule-based matched fewer than requested, generate questions for top remaining sentences that contain top keywords
    if len(faqs) < max_faqs:
        for sentence in sentences:
            # Check if this sentence is already used as an answer
            if any(f['answer'] == sentence for f in faqs):
                continue
            
            s_lower = sentence.lower()
            for kw, _ in keyword_scores:
                if re.search(r'\b' + re.escape(kw) + r'\b', s_lower):
                    q = f"What key information is provided about '{kw.title()}'?"
                    if q not in seen_questions:
                        seen_questions.add(q)
                        faqs.append({
                            "question": q,
                            "answer": sentence,
                            "rule": "Keyword Context Match"
                        })
                        break
            if len(faqs) >= max_faqs:
                break

    return {
        "success": True if faqs else False,
        "message": f"Successfully generated {len(faqs)} FAQ pairs." if faqs else "Could not generate meaningful FAQs from the provided text.",
        "faqs": faqs,
        "keywords": keyword_scores[:12],
        "total_sentences": len(sentences)
    }
