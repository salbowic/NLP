import pandas as pd
import subprocess
import numpy as np
import tempfile
import re
import argparse

def run_svm(params_file):
    result = subprocess.run(['python', 'main_svm.py', params_file], capture_output=True, text=True)
    output = result.stdout
    error_output = result.stderr
    
    if result.returncode != 0:
        print(f"Error running SVM script with parameters file {params_file}: {error_output}")
        return None
    
    lines = output.split('\n')
    metrics = {
        "training_time": None,
        "test_time": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "conf_matrix": None
    }

    for line in lines:
        if "Training time:" in line:
            metrics["training_time"] = float(line.split(":")[1].strip())
        elif "Test time:" in line:
            metrics["test_time"] = float(line.split(":")[1].strip())
        elif "Accuracy:" in line:
            metrics["accuracy"] = float(line.split(":")[1].strip())
        elif "Precision:" in line:
            metrics["precision"] = float(line.split(":")[1].strip())
        elif "Recall:" in line:
            metrics["recall"] = float(line.split(":")[1].strip())
        elif "F1 Score:" in line:
            metrics["f1_score"] = float(line.split(":")[1].strip())
        elif "Confusion Matrix:" in line:
            conf_matrix = []
            for i in range(2):
                matrix_line = lines[lines.index(line) + 1 + i].strip()
                conf_matrix.append(list(map(int, re.findall(r'\d+', matrix_line))))
            metrics["conf_matrix"] = np.array(conf_matrix)
    return metrics

def read_parameter_sets(file_path):
    parameter_sets = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            params = {}
            for param in line.strip().split(','):
                try:
                    key, value = param.split('=')
                    params[key.strip()] = value.strip()
                except ValueError:
                    print(f"Skipping invalid parameter entry: {param}")
                    continue
            parameter_sets.append(params)
    return parameter_sets

def main(params_file):
    parameter_sets = read_parameter_sets(params_file)
    
    results = []

    for params in parameter_sets:
        print(f"Running SVM with parameters: {params}")
        # Write parameters to a temporary file
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt') as temp_params:
            temp_params_path = temp_params.name
            for key, value in params.items():
                temp_params.write(f"{key}={value}\n")

        # Run the SVM script with the parameters file
        metrics = run_svm(temp_params_path)
        
        if metrics is None:
            metrics = {
                "training_time": "error",
                "test_time": "error",
                "accuracy": "error",
                "precision": "error",
                "recall": "error",
                "f1_score": "error",
                "conf_matrix": "error"
            }
        
        results.append({**params, **metrics})

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv('svm_results.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SVM with different parameter sets.')
    parser.add_argument('params_file', type=str, help='Path to the parameters file')
    args = parser.parse_args()
    
    main(args.params_file)