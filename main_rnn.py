import argparse
import numpy as np
from data_split import get_dataset_and_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf


class RNN_classifier:
    def __init__(self, buffer_size, batch_size, vocab_size, drop_out, dataset_name, nb_epochs) -> None:
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.VOCAB_SIZE = vocab_size
        self.DROP_OUT = drop_out

        self.dataset_name = dataset_name
        self.nb_epochs = nb_epochs
    
    def load_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = get_dataset_and_split(self.dataset_name)
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train.values, self.y_train.values))
        self.test_dataset =  tf.data.Dataset.from_tensor_slices((self.x_test.values, self.y_test.values))
        
        self.train_dataset = self.train_dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def evaluate_model(self):
        encoder = tf.keras.layers.TextVectorization(max_tokens=self.VOCAB_SIZE)
        encoder.adapt(self.train_dataset.map(lambda text, label: text))

        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(self.DROP_OUT),
            tf.keras.layers.Dense(1)
        ])

        model.summary()
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        history = model.fit(self.train_dataset, epochs=self.nb_epochs,
                    validation_data=self.test_dataset)

        test_loss, test_acc, test_precision, test_recall = model.evaluate(self.test_dataset)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)
        print('Test Precision:', test_precision)
        print('Test Recall:', test_recall) 

        # F1 Score
        test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print('Test F1 Score:', test_f1_score)

        # Predictions confusion matrix
        y_pred = model.predict(self.test_dataset)
        y_pred_labels = np.where(y_pred >= 0.5, 1, 0)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred_labels)
        print('Confusion Matrix:\n', conf_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNN Classifier Parameters')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Buffer size for shuffling the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size for the text vectorization')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout rate for the dropout layer')
    parser.add_argument('--dataset_name', type=str, default='twitter', help='Name of the dataset to use')
    parser.add_argument('--nb_epochs', type=int, default=5, help='Number of epochs for training the model')

    args = parser.parse_args()

    rc = RNN_classifier(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        drop_out=args.drop_out,
        dataset_name=args.dataset_name,
        nb_epochs=args.nb_epochs
    )
    rc.load_data()
    rc.evaluate_model()
