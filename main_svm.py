import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from thundersvm import SVC as ThunderSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import sys
from data_split import get_dataset_and_split
import time

def read_parameters(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                params[key.strip()] = value.strip()
    return params

def main(params_file):
    try:
        # Read parameters from the text file
        params = read_parameters(params_file)
        dataset_name = params.get('dataset_name', 'imdb')
        kernel = params.get('kernel', 'linear')
        C = float(params.get('C', 1.0))
        
        # Print the starting parameters
        print(f"SVM Parameters: dataset_name={dataset_name}, kernel={kernel}, C={C}")

        # Get the dataset
        x_train, x_test, y_train, y_test = get_dataset_and_split(dataset_name)
        
        # Convert text data to numerical features using TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        # Initialize SVM classifier
        svm_classifier = ThunderSVC(kernel=kernel, C=C, gpu_id=0)

        # Train SVM classifier
        start_fit_time = time.time()
        svm_classifier.fit(x_train, y_train)
        end_fit_time = time.time()
        fit_time = end_fit_time - start_fit_time

        # Predict sentiment on the test set
        start_predict_time = time.time()
        y_pred = svm_classifier.predict(x_test)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Training time: {fit_time}")
        print(f"Test time: {predict_time}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:")
        for row in conf_matrix:
            print(' '.join(map(str, row)))

    except Exception as e:
        print(f"Error running SVM: {e}")
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_svm.py <params_file>")
        sys.exit(1)
    params_file = sys.argv[1]
    main(params_file)
