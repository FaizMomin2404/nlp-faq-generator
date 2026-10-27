import re
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple, Set, Optional
from nlp_processor import tokenize_sentences, extract_keywords_tfidf

def clean_subject_phrase(phrase: str) -> str:
    """
    Cleans a matched subject phrase by removing leading articles, balanced/unbalanced parentheses, and punctuation.
    """
    s = phrase.strip()
    # Strip leading/trailing non-alphanumeric (except closing paren if opening is present)
    s = re.sub(r"^[^\w\(]+|[^\w\)]+$", "", s)
    # Strip leading determiners if present
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
    # Balance parentheses if unbalanced
    if s.count('(') > s.count(')'):
        s += ')'
    elif s.count(')') > s.count('('):
        s = s.replace(')', '')
    return s.strip()

def rule_definitional(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Matches definitional and classification sentences:
    e.g. "X is a Y", "X refers to Y", "X is defined as Y" -> "What is X?", "What does X refer to?"
    """
    s = sentence.rstrip('.!?')
    
    # "X refers to / means Y"
    m_ref = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,50}?)\s+(?:refers to|means)\s+(.+)", s, re.IGNORECASE)
    if m_ref:
        subject = clean_subject_phrase(m_ref.group(1))
        if 1 <= len(subject.split()) <= 6:
            return f"What does {subject} refer to?", "Definitional (Refers to)"

    # "X is/are defined as / known as Y"
    m_def = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,50}?)\s+(is|are)\s+(?:defined as|known as|described as)\s+(.+)", s, re.IGNORECASE)
    if m_def:
        subject = clean_subject_phrase(m_def.group(1))
        verb = m_def.group(2).lower()
        if 1 <= len(subject.split()) <= 6:
            return f"How {verb} {subject} defined?", "Definitional (Defined as)"

    # "X is/are a/an/the [Noun Phrase]"
    m_is_a = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,50}?)\s+(is|are)\s+(?:a|an|the)\s+([a-zA-Z0-9\s\-,]+)", s, re.IGNORECASE)
    if m_is_a:
        subject = clean_subject_phrase(m_is_a.group(1))
        verb = m_is_a.group(2).lower()
        if 1 <= len(subject.split()) <= 6:
            return f"What {verb} {subject}?", "Definitional (Is/Are)"

    return None

def rule_location_where(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Matches location and geographic sentences:
    e.g. "The college is located in Ashta, Maharashtra." -> "Where is the college located?"
    """
    s = sentence.rstrip('.!?')
    m_loc = re.match(r"^([a-zA-Z0-9\s\(\)\-]{2,50}?)\s+(is|are|was|were)\s+(located|situated|based|headquartered)\s+in\s+([a-zA-Z0-9\s,]+)", s, re.IGNORECASE)
    if m_loc:
        subject = clean_subject_phrase(m_loc.group(1))
        verb = m_loc.group(2).lower()
        loc_verb = m_loc.group(3).lower()
        if 1 <= len(subject.split()) <= 6:
            return f"Where {verb} {subject} {loc_verb}?", "Location (Where)"
    
    # "in [City/Country/Location], [Subject]..."
    m_in_loc = re.match(r"^(?:in|at)\s+([A-Z][a-zA-Z\s]+),\s+([a-zA-Z0-9\s\(\)\-]+?)\s+(is|are|was|were)\s+(.+)", s)
    if m_in_loc:
        subject = clean_subject_phrase(m_in_loc.group(2))
        verb = m_in_loc.group(3).lower()
        if 1 <= len(subject.split()) <= 5:
            return f"Where {verb} {subject}?", "Location (Where)"

    return None

def rule_temporal_when(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Matches time, year, and temporal origin sentences:
    e.g. "The university was established in 1983." -> "When was the university established?"
    """
    s = sentence.rstrip('.!?')
    m_time = re.match(r"^([a-zA-Z0-9\s\(\)\-]{2,50}?)\s+(was|were|is|are)\s+(founded|established|started|invented|created|introduced|released|launched|built|born)\s+(?:in|on|during)\s+([0-9]{4}|[a-zA-Z0-9\s,]+)", s, re.IGNORECASE)
    if m_time:
        subject = clean_subject_phrase(m_time.group(1))
        verb = m_time.group(2).lower()
        action = m_time.group(3).lower()
        if 1 <= len(subject.split()) <= 6:
            return f"When {verb} {subject} {action}?", "Temporal (When)"
    return None

def rule_person_who(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Matches personal attribution and leadership:
    e.g. "Dr. APJ Abdul Kalam was an Indian aerospace scientist who inspired millions of students."
         -> "Who was Dr. APJ Abdul Kalam?"
    """
    s = sentence.rstrip('.!?')
    
    # Honorific or personal name patterns
    m_person_was = re.match(r"^(?:The\s+)?((?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Sir\s+)?[A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+)+)\s+(was|is)\s+(?:a|an|the)\s+(.+)", s)
    if m_person_was:
        name = clean_subject_phrase(m_person_was.group(1))
        verb = m_person_was.group(2).lower()
        return f"Who {verb} {name}?", "Person (Who)"

    m_who_verb = re.match(r"^((?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Sir\s+)?[A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+)+)\s+(invented|founded|discovered|created|developed|designed|authored|led)\s+([a-zA-Z0-9\s\(\)\-]+)", s)
    if m_who_verb:
        action = m_who_verb.group(2).lower()
        object_phrase = clean_subject_phrase(m_who_verb.group(3))
        if 1 <= len(object_phrase.split()) <= 5:
            return f"Who {action} {object_phrase}?", "Person (Who)"
            
    return None

def rule_quantity_how_many(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Matches counts, numerical attributes, and quantities:
    e.g. "The engineering library contains over 50,000 books and reference journals."
         -> "How many books does the engineering library contain?"
    """
    s = sentence.rstrip('.!?')
    m_quant = re.match(r"^([a-zA-Z0-9\s\(\)\-]{2,40}?)\s+(has|have|contains|features|comprises|offers)\s+(?:over|more than|approximately|about|around)?\s*([0-9,]+|(?:two|three|four|five|six|seven|eight|nine|ten|hundred|thousand))\s+([a-zA-Z\s\-]+)", s, re.IGNORECASE)
    if m_quant:
        subject = clean_subject_phrase(m_quant.group(1))
        verb = m_quant.group(2).lower()
        verb_base = "contain" if "contain" in verb else ("have" if "ha" in verb else ("feature" if "feature" in verb else "offer"))
        raw_items = m_quant.group(4).strip()
        # Find head noun in the item chunk: split by space/conjunction
        item_chunk = re.split(r"\s+(?:and|or|with)\s+", raw_items)[0].strip()
        words = item_chunk.split()
        item = words[-1] if words else "items"
        # If the word is an adjective or non-noun, try full phrase
        if len(words) > 1 and len(words) <= 3:
            item = " ".join(words)
        if 1 <= len(subject.split()) <= 5:
            return f"How many {item} does {subject} {verb_base}?", "Quantity (How many)"
    return None

def rule_purpose_and_function(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Matches purpose, objective, or functional capabilities:
    e.g. "The primary objective of Natural Language Processing is to enable machines..."
    """
    s = sentence.rstrip('.!?')
    
    # "The objective/purpose/aim of X is to Y"
    m_purp = re.match(r"^(?:The\s+)?(primary\s+|main\s+|key\s+)?(objective|goal|purpose|aim|role|function)\s+of\s+([a-zA-Z0-9\s\(\)\-]+?)\s+(?:is|was)\s+to\s+(.+)", s, re.IGNORECASE)
    if m_purp:
        subject = clean_subject_phrase(m_purp.group(3))
        term = m_purp.group(2).lower()
        if 1 <= len(subject.split()) <= 6:
            return f"What is the {term} of {subject}?", "Purpose/Objective"

    # "X enables/allows/helps [target] to [action]"
    m_func = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+(enables|allows|helps|assists|permits|empowers)\s+(.+)", s, re.IGNORECASE)
    if m_func:
        subject = clean_subject_phrase(m_func.group(1))
        if 1 <= len(subject.split()) <= 6:
            return f"What does {subject} enable or help achieve?", "Functional (Capability)"

    # "X offers/provides/delivers Y"
    m_offer = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+(offers|provides|delivers|supports)\s+(.+)", s, re.IGNORECASE)
    if m_offer:
        subject = clean_subject_phrase(m_offer.group(1))
        verb = m_offer.group(2).lower()
        verb_base = "offer" if "offer" in verb else ("provide" if "provide" in verb else verb.rstrip('s'))
        if 1 <= len(subject.split()) <= 6:
            return f"What does {subject} {verb_base}?", "Functional (Service/Offer)"

    # "X is used for/to Y"
    m_used = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+is\s+(?:widely\s+|commonly\s+)?used\s+(?:to|for)\s+(.+)", s, re.IGNORECASE)
    if m_used:
        subject = clean_subject_phrase(m_used.group(1))
        if 1 <= len(subject.split()) <= 6:
            return f"What is {subject} used for?", "Usage (Application)"

    # "X is important because Y"
    m_bec = re.search(r"^([A-Z][a-zA-Z0-9\s\(\)\-]{2,40}?)\s+is\s+(important|essential|crucial|needed)\s+because\s+(.+)", s, re.IGNORECASE)
    if m_bec:
        subject = clean_subject_phrase(m_bec.group(1))
        adj = m_bec.group(2).lower()
        if 1 <= len(subject.split()) <= 6:
            return f"Why is {subject} considered {adj}?", "Causal (Why)"

    return None

def rule_pos_keyword_fallback(sentence: str, keywords_set: Set[str]) -> Optional[Tuple[str, str]]:
    """
    Fallback question generation using Part-Of-Speech (POS) tagging and TF-IDF keywords:
    Identifies candidate subject noun phrases and constructs interrogative forms.
    """
    s = sentence.rstrip('.!?')
    try:
        tokens = word_tokenize(s)
        tagged = nltk.pos_tag(tokens)
        
        # Look for leading noun phrases (e.g. NNP, NN, JJ NN)
        candidate_words = []
        for word, tag in tagged[:5]:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']:
                candidate_words.append(word)
            elif candidate_words and tag in ['VBZ', 'VBP', 'VBD', 'IN']:
                break
        
        if candidate_words and len(candidate_words) <= 4:
            cand_subject = clean_subject_phrase(" ".join(candidate_words))
            cand_lower = cand_subject.lower()
            
            # Check if this subject corresponds to an extracted domain keyword
            for kw in keywords_set:
                if kw in cand_lower or cand_lower in kw:
                    return f"What does the document state regarding {cand_subject}?", "POS/Keyword Association"
    except Exception:
        pass
    
    return None

def generate_faqs(text: str, max_faqs: int = 10) -> Dict:
    """
    Core FAQ Generation Pipeline:
    1. Sentence tokenization and normalization
    2. TF-IDF keyword extraction
    3. Linguistic rule-based question formulation across diverse question types:
       (What, Where, When, Who, How many, Why, How)
    4. Exact extractive answer sourcing (answers are strictly sourced from text)
    5. Deduplication, quality validation, and sorting
    """
    sentences = tokenize_sentences(text)
    if not sentences:
        return {
            "success": False,
            "message": "No complete sentences could be extracted from the document.",
            "faqs": [],
            "keywords": [],
            "total_sentences": 0
        }

    # Extract keywords using TF-IDF
    keyword_scores = extract_keywords_tfidf(sentences, top_n=20)
    keywords_set = {k[0].lower() for k in keyword_scores}

    faqs = []
    seen_questions = set()
    used_sentences = set()

    # Rule engines pipeline
    rule_pipeline = [
        rule_location_where,
        rule_temporal_when,
        rule_person_who,
        rule_quantity_how_many,
        rule_definitional,
        rule_purpose_and_function,
    ]

    # Pass 1: Apply specialized linguistic rules
    for sentence in sentences:
        for rule_fn in rule_pipeline:
            result = rule_fn(sentence)
            if result:
                q, rule_name = result
                norm_q = q.lower().strip()
                if norm_q not in seen_questions and sentence not in used_sentences:
                    seen_questions.add(norm_q)
                    used_sentences.add(sentence)
                    faqs.append({
                        "question": q,
                        "answer": sentence,
                        "rule": rule_name
                    })
                    break
        if len(faqs) >= max_faqs:
            break

    # Pass 2: Fallback with POS tagging and TF-IDF association for remaining capacity
    if len(faqs) < max_faqs:
        for sentence in sentences:
            if sentence in used_sentences:
                continue
            res = rule_pos_keyword_fallback(sentence, keywords_set)
            if res:
                q, rule_name = res
                norm_q = q.lower().strip()
                if norm_q not in seen_questions:
                    seen_questions.add(norm_q)
                    used_sentences.add(sentence)
                    faqs.append({
                        "question": q,
                        "answer": sentence,
                        "rule": rule_name
                    })
            if len(faqs) >= max_faqs:
                break

    # Pass 3: Keyword Context Match for sentences mentioning top keywords if still needed
    if len(faqs) < max_faqs:
        for sentence in sentences:
            if sentence in used_sentences:
                continue
            s_lower = sentence.lower()
            for kw, _ in keyword_scores:
                if re.search(r'\b' + re.escape(kw) + r'\b', s_lower):
                    q = f"What information is explained about '{kw.title()}'?"
                    norm_q = q.lower().strip()
                    if norm_q not in seen_questions:
                        seen_questions.add(norm_q)
                        used_sentences.add(sentence)
                        faqs.append({
                            "question": q,
                            "answer": sentence,
                            "rule": "Keyword Context Query"
                        })
                        break
            if len(faqs) >= max_faqs:
                break

    return {
        "success": True if faqs else False,
        "message": f"Successfully generated {len(faqs)} FAQ pairs." if faqs else "The source text did not contain sufficient factual patterns to generate FAQs.",
        "faqs": faqs,
        "keywords": keyword_scores[:12],
        "total_sentences": len(sentences)
    }
