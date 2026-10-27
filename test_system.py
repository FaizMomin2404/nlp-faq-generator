from document_loader import extract_text_from_txt, clean_text, validate_text
from faq_generator import generate_faqs

with open('sample_document.txt', 'rb') as f:
    raw = extract_text_from_txt(f.read())

cleaned = clean_text(raw)
valid, msg = validate_text(cleaned)
print("Valid:", valid)

res = generate_faqs(cleaned, max_faqs=10)
print("Success:", res["success"])
print("Extracted Keywords:")
for kw, score in res["keywords"][:6]:
    print(f"  - {kw}: {score:.2f}")

print(f"\nGenerated {len(res['faqs'])} FAQs:")
for idx, item in enumerate(res["faqs"], 1):
    print(f"\nFAQ #{idx} [{item['rule']}]:")
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
