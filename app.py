from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rake_nltk import Rake

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to summarize text
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=300, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to extract top 7 keywords
def extract_keywords(text):
    rake = Rake()  # Initialize RAKE for keyword extraction
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()  # Get ranked phrases
    return ', '.join(keywords[:7])  # Return the top 7 keywords

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    keywords = ""
    if request.method == 'POST':
        judgement_text = request.form['judgement_text']
        summary = summarize_text(judgement_text)
        keywords = extract_keywords(judgement_text)
    return render_template('index.html', summary=summary, keywords=keywords)

if __name__ == '__main__':
    app.run(debug=True)
