import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Step 1: Scraping the Wikipedia page
url = "https://en.wikipedia.org/wiki/Bangladesh"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
content = soup.get_text()

# Step 2: Use a pre-trained model to answer questions (used DistilBERT)
qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Step 3: Ask a question and get an answer from the model
question = input("Enter your question: ")
context = content
answer = answer_question(question, context)
print(f"Answer: {answer}")