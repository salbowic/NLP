import pandas as pd
from sklearn.model_selection import train_test_split
import re

## ================================
# Pliku uzywacie przez zaimportowanie z niego sobie funkcji get_dataset_and_split
## ================================

# funkcje clean_revire i read_valid_lines ogarniaja odczyt z pliku mcdonalda ktory jest nieco rozwalony
def clean_review(text):
    # Normalize unicode characters and replace new lines
    text = text.encode('ascii', errors='ignore').decode('utf-8')
    text = re.sub(r'[\r\n]+', ' ', text)
    return text.strip()

def read_valid_lines(filename):
    # Try reading the file with different encodings
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'Windows-1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"All encoding attempts failed for {filename}.")

    # Ensure that required columns exist
    if 'review' not in df.columns or 'rating' not in df.columns:
        raise ValueError("Required columns are missing from the data.")

    # Clean the review text
    df['review'] = df['review'].apply(clean_review)

    # Extract and map ratings to binary values
    df['rating'] = df['rating'].str.extract('(\d)')[0].astype(float)
    df['rating'] = df['rating'].map({1: 0, 2: 0, 4: 1, 5: 1})

    # Drop rows where 'rating' is NaN
    df.dropna(subset=['rating', 'review'], inplace=True)
    df['rating'] = df['rating'].astype(int)

    return df

def get_dataset_and_split(name, test_size_perc=0.2, rng_seed=1):
    df = None
    if name == "imdb":
        df = pd.read_csv("datasets/imdb.csv")
        # wywalenie z tekstu niepotrzebnych <br>
        df['review'] = df['review'].str.replace('<br />', '')
        # mapowanie slow sentymentu na wartosci 0 lub 1
        df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x == 'negative' else 1)
    elif name == "twitter":
        df = pd.read_csv("datasets/twitter.csv", names=["target", "ids", "date", "flag", "user", "text"], header=None)
        # wziecie tylko pozywynych = 4 lub negatywnych = 0 próbek
        df = df[df['target'].isin([0, 4])]
        # wybranie po 25k każdej
        df_pos = df[df['target'] == 4].sample(n=25000, random_state=rng_seed)
        df_neg = df[df['target'] == 0].sample(n=25000, random_state=rng_seed)
        df = pd.concat([df_pos, df_neg])
        # zmapowanie oryginalnych wartoci na nasze
        df['target'] = df['target'].map({0: 0, 4: 1})
        # ograniczenie do 2 potrzebnych nam kolumn
        df = df[['text', 'target']].rename(columns={'text': 'review', 'target': 'sentiment'})
        # naprawa zepsutego w oryginale kodowania znaków
        df['review'] = df['review'].str.replace('&lt;', '<')
        df['review'] = df['review'].str.replace('&quot;', '"')
        df['review'] = df['review'].str.replace('&amp;', '&')
        df['review'] = df['review'].str.replace('&gt;', '>')
    elif name == "mcdonald":
        df = read_valid_lines('datasets/mcdonald.csv')
        df.rename(columns={'rating': 'sentiment'}, inplace=True)
        df = df[['review', 'sentiment']]

    if df is not None:
        # sprawdzenie czy na pewno dobrze zmapowano sentyment
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0).astype(int)
        # podział danych na train i test tradycyjny
        x_train, x_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], stratify=df['sentiment'], test_size=test_size_perc, random_state=rng_seed)
        return x_train, x_test, y_train, y_test
    else:
        return None, None, None, None
    
# Example usage
# x_train, x_test, y_train, y_test = get_dataset_and_split("twitter")
# if x_train is not None:
#     print("Training labels count:\n", y_train.value_counts())
#     print("Test labels count:\n", y_test.value_counts())
#     print("Sample training labels:\n", y_train.head(5))
#     print("Sample training reviews:\n", x_train.head(5))
# else:
#     print("Failed to load and process the McDonald's data.")
