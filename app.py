# app.py
from flask import Flask, request, render_template, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline,BartTokenizer,BartForConditionalGeneration,AutoTokenizer
import torch
from textblob import TextBlob
from collections import Counter
import spacy
import logging
import joblib 

# Define the number of classes (topics) in your classification task
NUM_CLASSES = 4  # Update this based on your specific use case

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Load the fine-tuned DistilBERT model and tokenizer for text classification
model_classification = DistilBertForSequenceClassification.from_pretrained('/Users/manikmalhotra/Downloads/document_summarization/fine_tuned_model')
# Load the fine-tuned DistilBERT tokenizer for text classification
tokenizer_classification_path = '/Users/manikmalhotra/Downloads/document_summarization/fine_tuned_model/tokenizer'
tokenizer_classification = DistilBertTokenizer.from_pretrained(tokenizer_classification_path, local_files_only=True)

# Load the summarization model and tokenizer
tokenizer_summarization = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model_summarization = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Load the label_encoder
label_encoder_path = '/Users/manikmalhotra/Downloads/document_summarization/fine_tuned_model/label_encoder.joblib'
label_encoder = joblib.load(label_encoder_path)

# Configure error logging
logging.basicConfig(filename='/Users/manikmalhotra/Downloads/document_summarization/app.log', level=logging.ERROR)

# Define a function for document summarization
# Define a function for document summarization
# Inside the summarize_document function in app.py
# Inside the summarize_document function in app.py
def summarize_document(document):
    try:
        # Load the tokenizer for summarization
        tokenizer_summarization = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        # Tokenize the document
        inputs = tokenizer_summarization(document, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary using the model
        summary_ids = model_summarization.generate(**inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Convert the generated summary IDs to text
        summary = tokenizer_summarization.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    except Exception as e:
        # Add print statements or log messages to help identify the issue
        print(f"Error in summarize_document: {str(e)}")
        logging.error(f"Error in summarize_document: {str(e)}")
        return "Error in summarization. Please try again."



# Define a function for document truncation
def truncate_document(document, max_words=500):
    words = document.split()
    truncated_document = ' '.join(words[:max_words])
    return truncated_document

# Define a function for word frequency analysis
def word_frequency_analysis(document, top_n=10):
    words = document.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)
    return top_words

# Define a function for named entity recognition
def named_entity_recognition(document):
    doc = nlp(document)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities



# Define a function for text classification
# Define a function for text classification
# Define a function for text classification
# Define a function for text classification
def classify_document_with_fine_tuned_model(document, label_encoder=None):
    try:
        inputs = tokenizer_classification(document, return_tensors="pt", truncation=True)

        # Forward pass through the model
        outputs = model_classification(**inputs)

        # Get the predicted class probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy().tolist()[0]

        # Get the predicted class with the highest probability
        predicted_class_index = max(range(NUM_CLASSES), key=lambda i: probabilities[i])

        if label_encoder is not None:
            # Map the numerical class index to the original label
            predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        else:
            predicted_class_label = None

        return predicted_class_label, probabilities

    except Exception as e:
        # Add print statements or log messages to help identify the issue
        print(f"Error in classify_document_with_fine_tuned_model: {str(e)}")
        logging.error(f"Error in classify_document_with_fine_tuned_model: {str(e)}")
        return "Error in text classification. Please try again.", []


@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        document = data['document']

        # Classify the document
        predicted_class_label, probabilities = classify_document_with_fine_tuned_model(document, label_encoder)

        response_data = {
            'predicted_class': predicted_class_label,
            'class_probabilities': probabilities
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in text classification: {str(e)}")
        return jsonify({'error': 'Error in text classification. Please try again.'})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        document = data['document']

        # Truncate document to 500 words
        truncated_document = truncate_document(document)

        summary = summarize_document(truncated_document)

        # Word frequency analysis
        word_freq = word_frequency_analysis(document)

        # Named Entity Recognition
        entities = named_entity_recognition(document)

        # Document Length Information
        word_count = len(document.split())
        char_count = len(document)

        response_data = {
            'summary': summary,
            'word_freq': word_freq,
            'entities': entities,
            'word_count': word_count,
            'char_count': char_count
        }

        return jsonify(response_data)

    except Exception as e:
        # Log the error
        logging.error(f"Error in summarization: {str(e)}")
        return jsonify({'error': 'Error in summarization. Please try again.'})

@app.route('/classify', methods=['POST'])
def text_classify():
    try:
        data = request.get_json()
        document = data['document']

        # Classify the document
        predicted_class_label, probabilities = classify_document_with_fine_tuned_model(document, label_encoder)

        response_data = {
            'predicted_class': predicted_class_label,
            'class_probabilities': probabilities
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in text classification: {str(e)}")
        return jsonify({'error': 'Error in text classification. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
