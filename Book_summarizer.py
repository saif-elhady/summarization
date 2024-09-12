# Install necessary libraries
import pdfplumber

# Function to convert PDF to text using pdfplumber
def pdf_to_text_plumber(pdf_path):
    text = ''

    # Open and read PDF file page by page
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    return text

# Provide the path to your PDF file
pdf_path = r"D:\my_env\Aurelien_Geron_Hands_On_Machine_Learning_with_Scikit_Learn_Keras.pdf"

# Extract the text from the PDF
book_text = pdf_to_text_plumber(pdf_path)

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained Sentence-BERT model for semantic embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def divide_by_semantics_with_length(text, threshold=0.6, max_words=1000, min_words=400):
    # Split text into sentences for semantic comparison
    sentences = text.split('. ')  # Simple sentence splitting based on periods. Adjust if needed.
    
    # Encode all sentences into embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # List to hold the different chunks based on semantic shifts
    chunks = []
    current_chunk = sentences[0]  # Start with the first sentence
    
    for i in range(1, len(sentences)):
        # Calculate cosine similarity between current sentence and the previous one
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i-1])

        # Check word count of current chunk
        current_word_count = len(current_chunk.split())

        # If similarity drops below the threshold or the current chunk exceeds max_words, start a new chunk
        if similarity < threshold or current_word_count + len(sentences[i].split()) > max_words:
            # Ensure the chunk meets the minimum word count requirement
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())  # Add the current chunk to the list
                current_chunk = sentences[i]  # Start a new chunk
            else:
                # If below min_words, keep adding to the current chunk
                current_chunk += '. ' + sentences[i]
        else:
            current_chunk += '. ' + sentences[i]  # Continue with the current chunk
    
    # Append the last chunk (ensure it meets the minimum length condition)
    if len(current_chunk.split()) >= min_words:
        chunks.append(current_chunk.strip())
    
    return chunks

# Divide the book text into semantic chunks
semantic_chunks = divide_by_semantics_with_length(book_text)

import re
from nltk.tokenize import sent_tokenize

# Function to clean text by removing unwanted characters
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Clean all chunks of the book
def clean_chunks(chunks):
    return [clean_text(chunk) for chunk in chunks]

# Clean the semantic chunks
cleaned_semantic_chunks = clean_chunks(semantic_chunks)

from transformers import pipeline

# Load a DistilBART-based summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # Use CPU

# Define maximum tokens DistilBART can handle (usually 1024 tokens)
MAX_TOKEN_LENGTH = 1024

def summarize_chunks(chunks):
    summaries = []
    for chunk in chunks:
        chunk_length = len(chunk.split())

        if chunk_length > 20:  # Only summarize if chunk length is sufficient
            try:
                # Ensure that the chunk size is within DistilBART's token limit
                summary = summarizer(chunk, max_length=800, min_length=1, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summaries.append(chunk)  # Fallback: use original chunk if summarization fails
        else:
            summaries.append(chunk)  # Keep the chunk as is if it's too short

    return summaries

# Summarize the cleaned chunks
summarized_chunks = summarize_chunks(cleaned_semantic_chunks)


# Combine summaries into one final overall summary
def overall_summary(summaries):
    return ' '.join(summaries)

# Create the final overall summary
final_summary = overall_summary(summarized_chunks)

from fpdf import FPDF

# Function to replace or remove non-latin-1 characters
def strip_unicode(text):
    return text.encode('latin-1', 'ignore').decode('latin-1')

# Create a PDF class inheriting from FPDF
class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:  # Ensure the header is only on the first page
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Book Summary', ln=True, align='C')
            self.ln(10)  # Add a bit of space after the header
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_text(self, text):
        self.add_page()
        self.chapter_body(text)

# Create a PDF instance
pdf = PDF()

# Clean the final summary to remove any non-latin-1 characters for PDF compatibility
cleaned_summary = strip_unicode(final_summary)

# Add the cleaned summary text to the PDF
pdf.add_text(cleaned_summary)

import os

# Ensure the directory exists
output_dir = "D:/summary__files"
#os.makedirs(output_dir, exist_ok=True)

# Define the full file path
file_path = os.path.join(output_dir, "summary.pdf")

# Save the PDF
pdf.output(file_path)


# Save the PDF to a specified file path
#file_path = "D:\summary"
pdf.output(file_path)

# Display the file path (optional)
file_path
