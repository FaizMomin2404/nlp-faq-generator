import pymupdf

doc = pymupdf.open()
page = doc.new_page(width=595, height=842) # A4 size

rect = pymupdf.Rect(50, 50, 545, 792)

content = (
    "Machine Learning and Deep Learning Concepts.\n\n"
    "The research center is located in Pune, India.\n"
    "The institution was established in 1995.\n"
    "Dr. Alan Turing was an English mathematician and pioneer in computing.\n"
    "The computing laboratory contains over 100 high-performance servers.\n\n"
    "Supervised learning is a machine learning paradigm where models are trained using labeled datasets.\n"
    "The primary objective of supervised learning is to predict discrete outcomes or continuous values accurately.\n"
    "Convolutional Neural Networks are widely used for computer vision and image classification tasks.\n"
    "Feature engineering is important because high quality representations significantly improve prediction accuracy.\n"
)

page.insert_textbox(rect, content, fontsize=11, lineheight=1.4)
pdf_path = "sample_test.pdf"
doc.save(pdf_path)
doc.close()
print(f"Created {pdf_path} using textbox successfully!")
