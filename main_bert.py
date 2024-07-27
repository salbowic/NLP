import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import AdamW
from tqdm import tqdm
import torchmetrics
import json
from data_split import get_dataset_and_split
import time
import argparse

# Create the parser and arguments
parser = argparse.ArgumentParser(description="Set parameters for the training model.")
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
parser.add_argument('--review_type_name', type=str, default='imdb', help='Type of reviews (e.g., "twitter", "imdb").')
parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length, default is 256.')
parser.add_argument('--version', type=int, default=1, help='Version number of the model.')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for the optimizer.')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for the BERT model.')

args = parser.parse_args()

# folders for saving
cache_dir = "/home/karol/NLP_models"
model_dir = "bert_models"
metrics_dir = "bert_metrics"

# model bert base uncased and caching dir for models
model_name = 'bert-base-uncased'
review_type_name = args.review_type_name

# Hyperparameters
max_length = args.max_length
batch_size = args.batch_size
epochs = args.epochs
version = args.version
learn_rate = args.lr
dropout_rate = args.dropout_rate
train_shuffle = True

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loading data and printing basic info about the dataset
x_train, x_test, y_train, y_test = get_dataset_and_split(review_type_name)
print("Training labels count:\n", y_train.value_counts())
print("Test labels count:\n", y_test.value_counts())
print("Sample training labels:\n", y_train.head(5))
print("Sample training reviews:\n", x_train.head(5))

# Initialize tokenizer and encode
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

def encode_reviews(tokenizer, reviews, max_length):
    return tokenizer(reviews.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

train_encodings = encode_reviews(tokenizer, x_train, max_length)
test_encodings = encode_reviews(tokenizer, x_test, max_length)

# Prepare datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train.values))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test.values))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# Load BERT model and optimizer
config = BertConfig.from_pretrained(model_name, cache_dir=cache_dir, hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate, num_labels=2)
model = BertForSequenceClassification.from_pretrained(model_name, config=config).to(device)
optimizer = AdamW(model.parameters(), lr=learn_rate)

# Metrics
accuracy_metric = torchmetrics.Accuracy(num_classes=2, task='binary').to(device)
precision_metric = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)
recall_metric = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
conf_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=2, task='binary').to(device)

metrics_history = {
    'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_F1': [], 'train_confusion_matrix': [], 'train_time': [],
    'test_accuracy': [], 'test_precision': [], 'test_recall': [], 'test_F1': [], 'test_confusion_matrix': [], 'test_time': []
}

# Print data about hyperparameters
print("Current parameters:")
print(f"Dataset name: {review_type_name}")
print(f"Batch: {batch_size}")
print(f"Length: {max_length}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {learn_rate}")
print(f"Dropout rate: {dropout_rate}")
print(f"Version: {version}")
save_name = f'{review_type_name}_model_batch_{batch_size}_max_{max_length}_epoch_{epochs}_lr_{learn_rate}_dr_{int(dropout_rate*100)}_v{version}.pt'
print(f"Save name: {save_name}")

# Main epochs loop
for epoch in range(epochs):
    # Training loop with metrics
    model.train() 
    epoch_loss = 0.0
    progress_bar_train = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False)
    epoch_start_time = time.time()

    # main loop for train data
    for batch in progress_bar_train:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids, b_input_mask, b_labels = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy_metric.update(preds, b_labels)
        precision_metric.update(preds, b_labels)
        recall_metric.update(preds, b_labels)
        conf_matrix_metric.update(preds, b_labels)
        epoch_loss += loss.item()

    # Compute Metrics for TRAIN DATASET
    train_epoch_time = time.time() - epoch_start_time
    train_acc = accuracy_metric.compute()
    train_prec = precision_metric.compute()
    train_recall = recall_metric.compute()
    train_f1 = 2 * (train_prec * train_recall) / (train_prec + train_recall)
    train_conf_matrix = conf_matrix_metric.compute()
    # Save metrics for TRAIN DATASET
    metrics_history['train_loss'].append(epoch_loss / len(train_loader))
    metrics_history['train_accuracy'].append(train_acc.item())
    metrics_history['train_precision'].append(train_prec.item())
    metrics_history['train_recall'].append(train_recall.item())
    metrics_history['train_F1'].append(train_f1.item())
    metrics_history['train_confusion_matrix'].append(train_conf_matrix.tolist())
    metrics_history['train_time'].append(train_epoch_time)  # Save training time
    # Print current metric for TRAIN DATASET
    print(f"Epoch {epoch + 1} - Training - Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}, Conf Matrix: {train_conf_matrix.tolist()}")

    # Reset training metrics after TRAIN DATASET TRAIN
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    conf_matrix_metric.reset()

    # Test evaluation
    model.eval()
    progress_bar_test = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}/{epochs}", leave=False)

    eval_start_time = time.time()

    # main loop for eval data
    with torch.no_grad():
        for batch in progress_bar_test:
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            accuracy_metric.update(preds, b_labels)
            precision_metric.update(preds, b_labels)
            recall_metric.update(preds, b_labels)
            conf_matrix_metric.update(preds, b_labels)

    # Compute Metrics for TEST DATASET
    test_epoch_time = time.time() - eval_start_time
    test_accuracy = accuracy_metric.compute()
    test_precision = precision_metric.compute()
    test_recall = recall_metric.compute()
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    test_conf_matrix = conf_matrix_metric.compute()
    # Save metrics for TEST DATASET
    metrics_history['test_accuracy'].append(test_accuracy.item())
    metrics_history['test_precision'].append(test_precision.item())
    metrics_history['test_recall'].append(test_recall.item())
    metrics_history['test_F1'].append(test_f1.item())
    metrics_history['test_confusion_matrix'].append(test_conf_matrix.tolist())
    metrics_history['test_time'].append(test_epoch_time)  # Save evaluation time
    # Print current metric for TEST DATASET
    print(f"Epoch {epoch + 1} - Test     - Loss: N/A   , Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}, Conf Matrix: {test_conf_matrix.tolist()}")

    # Reset evaluation metrics after TEST DATASET EVAL
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    conf_matrix_metric.reset()

# Save the model
torch.save(model.state_dict(), f"{model_dir}/{save_name}")

# Save metrics history
metrics_file_name = f"{metrics_dir}/metrics_{save_name[:-3]}.json"
with open(metrics_file_name, 'w') as file:
    json.dump(metrics_history, file)
