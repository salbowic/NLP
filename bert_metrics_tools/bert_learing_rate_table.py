import tkinter as tk
from tkinter import filedialog
import json
import re

# Data structure to hold the extracted metrics
metrics_data = []

def extract_learning_rate(filename):
    # Using regex to find the learning rate in the filename
    match = re.search(r"lr_([1-9]e-\d+)_", filename)
    if match:
        return match.group(1)
    return "2e-05"  # Default learning rate if not found


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
        learning_rates = sorted(set(row[0] for row in metrics_data))  # Extract unique learning rates

        # Writing each metric for each learning rate
        for lr in learning_rates:
            file.write(f"\\multicolumn{{7}}{{c}}{{Learning Rate = $\\num{{{lr}}}$}} \\\\ \\hline\n")
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
                    values = [f"${{\\num{{{v:.0f}}}}}$" if metric == "Time" else f"${{\\num{{{v:.4f}}}}}$" for v in extract_values(lr, metric, dataset)]
                    file.write(metric_label + " & " + ' & '.join(values) + " \\\\\n")
                file.write("\\hline\n")  # Add horizontal line after each dataset block
        file.write('\\end{tabular}\n')

def extract_values(learning_rate, metric, dataset):
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
    return [row[index] for row in metrics_data if row[0] == learning_rate]

def append_metrics_to_table(data, learning_rate):
    # Extract the needed metrics and format them for LaTeX table row; reformatted for new table design
    metrics = [
        data['train_accuracy'],
        data['train_F1'],
        data['train_time'],
        data['test_accuracy'],
        data['test_F1']
    ]
    
    # Creating rows with learning rate and metrics; each metric will now appear in its own row
    for metric_values in zip(*metrics):  # Transposes the list to group by epoch
        row = [learning_rate] + list(metric_values)
        metrics_data.append(row)

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filename = filedialog.askopenfilename(title='Select a JSON file with metrics')
    if filename:
        learning_rate = extract_learning_rate(filename)
        data = load_metrics(filename)
        append_metrics_to_table(data, learning_rate)
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
