import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Set

# Initialize required NLTK resources
def ensure_nltk_resources():
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

ensure_nltk_resources()

def tokenize_sentences(text: str) -> List[str]:
    """
    Splits text into meaningful, clean sentences.
    Filters out overly short sentence fragments, titles, or page numbers.
    """
    raw_sentences = sent_tokenize(text)
    clean_sentences = []
    
    for s in raw_sentences:
        s_clean = s.strip()
        # Remove repeated internal spaces
        s_clean = re.sub(r"\s+", " ", s_clean)
        # Check minimum word length and presence of alphabetical characters
        words = s_clean.split()
        if len(words) >= 5 and any(c.isalpha() for c in s_clean):
            clean_sentences.append(s_clean)
            
    return clean_sentences

def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes text into words and alphanumeric tokens.
    """
    return word_tokenize(text)

def get_stop_words() -> Set[str]:
    """
    Returns an enriched set of English stop words tailored for keyword extraction.
    """
    try:
        base_stops = set(stopwords.words('english'))
    except Exception:
        base_stops = {'the', 'a', 'an', 'is', 'are', 'in', 'on', 'of', 'and', 'for', 'to', 'it', 'this', 'that'}
        
    additional_stops = {
        'use', 'uses', 'using', 'used', 'also', 'many', 'much', 'like', 'such',
        'one', 'two', 'three', 'can', 'may', 'could', 'would', 'shall', 'might',
        'often', 'well', 'etc', 'e.g', 'i.e', 'given', 'based', 'every', 'within'
    }
    return base_stops.union(additional_stops)

def extract_keywords_tfidf(sentences: List[str], top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Extracts informative keywords and key phrases using TF-IDF across sentences.
    Returns sorted list of (keyword, score) tuples.
    """
    if not sentences:
        return []

    stop_words_list = list(get_stop_words())

    try:
        # Use unigrams and bigrams for domain concepts
        vectorizer = TfidfVectorizer(
            stop_words=stop_words_list,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b',
            max_df=0.85,
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        
        keyword_scores = []
        for name, score in zip(feature_names, scores):
            # Exclude hyphen-only or purely numerical artifacts
            if name.replace('-', '').isalpha() and score > 0.05:
                keyword_scores.append((name, float(score)))

        # Sort descending by TF-IDF score
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate overlapping unigram-bigram (e.g. keep 'natural language processing' if 'language' is lower)
        seen = set()
        deduped = []
        for kw, score in keyword_scores:
            if kw not in seen:
                seen.add(kw)
                deduped.append((kw, score))
                if len(deduped) >= top_n:
                    break

        return deduped
    except Exception:
        return []
