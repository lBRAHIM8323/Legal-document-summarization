from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rake_nltk import Rake
import fitz  # PyMuPDF

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to summarize text
# Function to summarize text
def summarize_text(text, max_length):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Adjust max_length to the number of tokens corresponding to 500 words
    # Generally, 1 word ≈ 1.3 tokens in T5
    max_token_length = min(max_length, 500 * 1.3)

    summary_ids = model.generate(
        inputs,
        max_length=int(max_token_length),
        min_length=min(200, max_length - 100),
        length_penalty=1.0,
        num_beams=2,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to extract top 7 keywords
def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return ', '.join(keywords[:7])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with fitz.open(pdf_file.stream) as doc:
            for page in doc:
                text += page.get_text()
        if not text:
            print("No text extracted from PDF.")
    except Exception as e:
        print("Error extracting text from PDF:", e)
    return text

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    keywords = ""
    judgement_text = ""  # Initialize judgement_text variable

    if request.method == 'POST':
        judgement_text = request.form.get('judgement_text', '')  # Get text from the form
        summary_length = int(request.form['summary_length'])

        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename != '':
                print("PDF file received:", pdf_file.filename)
                pdf_text = extract_text_from_pdf(pdf_file)
                judgement_text = pdf_text if pdf_text else judgement_text
        
        summary = summarize_text(judgement_text, summary_length)
        keywords = extract_keywords(judgement_text)

    return render_template('index.html', summary=summary, keywords=keywords, judgement_text=judgement_text)

if __name__ == '__main__':
    app.run(debug=True)
