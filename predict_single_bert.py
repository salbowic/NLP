import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse
import re

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict sentiment of a review using BERT model")
parser.add_argument("--model_name", type=str, help="Model name to load (folder must be named accordingly)")
parser.add_argument("--text_analyze", type=str, help="Review text to analyze")
args = parser.parse_args()

# input parameters
model_name = args.model_name
text_analyze = args.text_analyze

# Extract max_length from model name
max_length_match = re.search(r"max_(\d+)", model_name)
if max_length_match:
    max_length = int(max_length_match.group(1))
else:
    raise ValueError("Model name does not contain max length information.")

# Load the PyTorch model
model_path = f'bert_models/{model_name}.pt'  # adjust this path as necessary
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_review(tokenizer, review, max_length):
    return tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

def predict_sentiment(review, max_length):
    encoded_review = encode_review(tokenizer, review, max_length)
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    sentiment = torch.argmax(logits, axis=1).item()
    return "Positive" if sentiment == 1 else "Negative"

if __name__ == "__main__":
    sentiment = predict_sentiment(text_analyze, max_length)
    print(f"Sentiment of the review: {sentiment}")
