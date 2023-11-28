# Document Summarization and Classification App

This repository contains the source code for a web application that performs document summarization and classification. The app leverages pre-trained natural language processing models for summarizing and classifying text documents.

## Features

- **Summarization:**
  - Utilizes the BART (Facebook's Bart) model for document summarization.
  - Provides a concise summary of input text.

- **Classification:**
  - Incorporates a fine-tuned DistilBERT model for text classification.
  - Classifies input documents into predefined categories.

## Usage

### Prerequisites

- Python 3.x
- Flask
- Transformers
- Torch
- SpaCy
- TextBlob

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manik997/nlp_text_summarization/tree/main
   
   cd nlp_text_summarization/tree/main
   
2. Install Dependencies
   pip install -r requirements.txt

3. Run the Flask app
   python app.py
   
Files like sports.txt,politics.txt,entertainment.txt are all testing data
Fine tuning is done on news_dataa.csv 
