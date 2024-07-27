import os
import json
import glob

def find_top_files(folder_path):
    results = {
        'imdb': [],
        'mcdonald': [],
        'twitter': []
    }
    
    # Pattern to match all json files in the folder
    file_pattern = os.path.join(folder_path, '*.json')
    files = glob.glob(file_pattern)
    
    # Process each file
    for file_path in files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            test_accuracy = max(data['test_accuracy'])
            
            # Determine the category of the model based on the filename
            if 'imdb' in file_path:
                category = 'imdb'
            elif 'mcdonald' in file_path:
                category = 'mcdonald'
            elif 'twitter' in file_path:
                category = 'twitter'
            else:
                continue  # Skip files that don't match any category
            
            # Append the result as a tuple (max accuracy, file name)
            results[category].append((test_accuracy, os.path.basename(file_path)))
    
    # Sort and get the top 5 for each category
    for key in results.keys():
        results[key] = sorted(results[key], reverse=True, key=lambda x: x[0])[:5]
    
    return results

def format_results(results):
    formatted_string = "Top files by category based on maximum 'test_accuracy':\n"
    for category, files in results.items():
        formatted_string += f"\nCategory: {category}\n"
        for accuracy, file in files:
            formatted_string += f"Accuracy: {accuracy:.4f} - File: {file}\n"
    return formatted_string

# Replace 'your_folder_path' with the path to your folder containing the JSON files
folder_path = '../bert_metrics'
top_files = find_top_files(folder_path)
formatted_output = format_results(top_files)
print(formatted_output)
