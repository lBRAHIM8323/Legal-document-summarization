from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pdfplumber  # For extracting text from PDFs
from flask_caching import Cache

# Initialize Flask app
app = Flask(__name__)

# Configure Flask-Caching
app.config['CACHE_TYPE'] = 'simple'  # Simple in-memory caching
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout in seconds
cache = Cache(app)

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('final')
tokenizer = T5Tokenizer.from_pretrained('final')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() + "\n"
    return pdf_text

# Function to summarize text
@cache.memoize()  # Cache this function to avoid reprocessing the same input
def summarize_text(text, max_length):
    if not text.strip():
        return "No text extracted from PDF to summarize."

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=500, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=min(max_length, 250),
        min_length=50,
        length_penalty=1.0,
        num_beams=2,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    judgement_text = ""

    if request.method == 'POST':
        judgement_text = request.form.get('judgement_text', '')
        summary_length = int(request.form['summary_length'])

        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename != '':
                print("PDF file received:", pdf_file.filename)
                pdf_text = extract_text_from_pdf(pdf_file)
                judgement_text = pdf_text if pdf_text else judgement_text

        # Summarize the text
        summary = summarize_text(judgement_text, summary_length)

    return render_template('index.html', summary=summary, judgement_text=judgement_text)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
