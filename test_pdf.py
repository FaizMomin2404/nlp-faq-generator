from document_loader import extract_text_from_pdf, clean_text, validate_text
from faq_generator import generate_faqs

with open('sample_test.pdf', 'rb') as f:
    pdf_bytes = f.read()

raw = extract_text_from_pdf(pdf_bytes)
cleaned = clean_text(raw)
valid, msg = validate_text(cleaned)
print("PDF Valid:", valid)

res = generate_faqs(cleaned, max_faqs=8)
print("PDF FAQ Count:", len(res["faqs"]))
for i, faq in enumerate(res["faqs"], 1):
    print(f"{i}. [{faq['rule']}] {faq['question']}")
    print(f"   Answer: {faq['answer']}")
