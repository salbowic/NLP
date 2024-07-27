import tkinter as tk
from tkinter import filedialog
import json
import re

# Data structure to hold the extracted metrics
metrics_data = []

def extract_batch_size(filename):
    # Using regex to find the batch size in the filename
    match = re.search(r"batch_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None

def load_metrics(filename):
    # Open and load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def save_table_to_file():
    # Convert the data into LaTeX tabular format and save to a file
    with open('metrics_output.txt', 'w') as file:
        # Writing the header with epochs indexed
        header = "Dataset & Metric & Epoch 1 & Epoch 2 & Epoch 3 & Epoch 4 & Epoch 5 \\\\ \\hline\n"
        file.write('\\begin{tabular}{c|c|ccccc}\n')
        file.write(header)

        # Available metrics and datasets
        metrics = {
            'Train': ["Accuracy", "F1", "Time"],
            'Test': ["Accuracy", "F1"]
        }
        batch_sizes = sorted(set(row[0] for row in metrics_data))  # Extract unique batch sizes

        # Writing each metric for each batch size
        for batch_size in batch_sizes:
            file.write("\\multicolumn{7}{c}{Batch Size = " + str(batch_size) + "} \\\\ \\hline\n")
            for dataset in ['Train', 'Test']:
                num_metrics = len(metrics[dataset])
                first_metric = True
                for metric in metrics[dataset]:
                    if first_metric:
                        file.write("\\multirow{" + str(num_metrics) + "}{*}{" + dataset + "} & ")
                        first_metric = False
                    else:
                        file.write("& ")  # Continue without adding dataset name again
                    metric_label = "F1 Score" if metric == "F1" else "Accuracy" if metric == "Accuracy" else "Time"
                    values = [f"${{\\num{{{v:.0f}}}}}$" if metric == "Time" else f"${{\\num{{{v:.4f}}}}}$" for v in extract_values(batch_size, metric, dataset)]
                    file.write(metric_label + " & " + ' & '.join(values) + " \\\\\n")
                file.write("\\hline\n")  # Add horizontal line after each dataset block
        file.write('\\end{tabular}\n')


def extract_values(batch_size, metric, dataset):
    # Adjust index based on the metric and dataset
    index = 0
    if dataset == "Train":
        if metric == "Accuracy":
            index = 1
        elif metric == "F1":
            index = 2
        elif metric == "Time":
            index = 3
    elif dataset == "Test":
        if metric == "Accuracy":
            index = 4
        elif metric == "F1":
            index = 5
    return [row[index] for row in metrics_data if row[0] == batch_size]


def append_metrics_to_table(data, batch_size):
    # Extract the needed metrics and format them for LaTeX table row; reformatted for new table design
    metrics = [
        data['train_accuracy'],
        data['train_F1'],
        data['train_time'],
        data['test_accuracy'],
        data['test_F1']
    ]
    
    # Creating rows with batch size and metrics; each metric will now appear in its own row
    for metric_values in zip(*metrics):  # Transposes the list to group by epoch
        row = [batch_size] + list(metric_values)
        metrics_data.append(row)

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filename = filedialog.askopenfilename(title='Select a JSON file with metrics')
    if filename:
        batch_size = extract_batch_size(filename)
        if batch_size is None:
            print("Batch size not found in filename.")
            return True
        data = load_metrics(filename)
        append_metrics_to_table(data, batch_size)
        return True
    else:
        return False

def main():
    continue_loading = True
    while continue_loading:
        continue_loading = open_file_dialog()
    save_table_to_file()
    print("Metrics table saved in 'metrics_output.txt'.")

if __name__ == "__main__":
    main()
