from data_preparation.prepare_dataset import prepare_log_dataset
from models.model_zoo import random_forest_model, logistic_regression_model, svm_model, lstm_model
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ML Model Training Function
def train_model(model, X_train, y_train):
    """Train the model."""
    logging.info(f"Training model: {model.__class__.__name__}")
    model.fit(X_train, y_train)
    logging.info(f"Model {model.__class__.__name__} trained successfully")

    # save the model
    model_file = f'../data/models/{model.__class__.__name__.lower()}_model.pkl'
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

    return model

# LSTM Model Training Function
def train_lstm_model(model, X_train, y_train, epochs, batch_size, validation_split):
    """Train the LSTM model."""
    logging.info(f"Training LSTM model: {model.__class__.__name__} with epochs={epochs}, batch_size={batch_size}, validation_split={validation_split}")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    logging.info(f"LSTM model {model.__class__.__name__} trained successfully")

    # Save train and validiation loss and accuracy plots to a file
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Loss vs Epoch')
    plt.plot(model.history.history['loss'], label='train', color='blue', marker='o')
    plt.plot(model.history.history['val_loss'], label='validation', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy vs Epoch')
    plt.plot(model.history.history['accuracy'], label='train', color='blue', marker='o')
    plt.plot(model.history.history['val_accuracy'], label='validation', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Title
    plt.suptitle(f'LSTM Model Training Loss and Accuracy')
    plt.tight_layout(pad=2.0)  # Add spacing between the two plots
    plt.savefig('../data/plots/lstm_loss_accuracy.png')   

    # Save the model
    model.save('../data/models/lstm_model.h5') 
    return model

# Load the dataset
output_dir = '../data/parsed_data/'
output_file = 'parsed_data.parquet'

# Dataset Preparation parameters
window_size = 32
window_overlap = 0.5
event_id_column = 'EventIds'
event_message_column = 'LogWindows'
max_features = 2000
dataset_output_dir = '../data/dataset/'
label_file_path = '../data/Hadoop/anamoly_label.csv'

# Prepare the dataset
logging.info("Preparing the dataset")
X_train, X_test, y_train, y_test = prepare_log_dataset(output_dir, output_file, window_size, window_overlap, label_file_path, event_id_column, event_message_column, max_features, dataset_output_dir)
logging.info("Dataset prepared successfully")

# Label Encoding for the labels
logging.info("Encoding labels")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
logging.info("Labels encoded successfully")

# Train the models
models = [random_forest_model(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1),
          logistic_regression_model(penalty='l2', C=1.0, solver='liblinear'),
          svm_model(C=1.0, kernel='rbf', degree=3, gamma='scale'),
          lstm_model(input_shape=(1, X_train.shape[1]), hidden_units=128, output=4)]

# Results Dictionary
results = {}

for model in models:
    logging.info(f"Starting training for model: {model.__class__.__name__}")
    if 'functional' in model.__class__.__name__.lower():
        X_train_expanded = np.expand_dims(X_train, axis=1)
        X_test_expanded = np.expand_dims(X_test, axis=1)
        model = train_lstm_model(model, X_train_expanded, y_train, epochs=20, batch_size=32, validation_split=0.2)
    else:
        model = train_model(model, X_train, y_train)

    # Evaluate the model
    logging.info(f"Evaluating model: {model.__class__.__name__}")
    if 'functional' in model.__class__.__name__.lower():
        y_pred = model.predict(X_test_expanded)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate the accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if 'functional' in model.__class__.__name__.lower():
        logging.info(f"Model: LSTM")
        logging.info(f"Accuracy: {accuracy:.3f}")
        logging.info(f"Precision: {precision:.3f}")
        logging.info(f"Recall: {recall:.3f}")
        logging.info(f"F1-Score: {f1:.3f}")
    else:    
        logging.info(f"Model: {model.__class__.__name__}")
        logging.info(f"Accuracy: {accuracy:.3f}")
        logging.info(f"Precision: {precision:.3f}")
        logging.info(f"Recall: {recall:.3f}")
        logging.info(f"F1-Score: {f1:.3f}")

    if 'functional' in model.__class__.__name__.lower():
        results["LSTM"] = {'Accuracy': round(accuracy, 2), 'Precision': round(precision, 2), 'Recall': round(recall, 2), 'F1-Score': round(f1, 2)}
    else:
        results[model.__class__.__name__] = {'Accuracy': round(accuracy, 2), 'Precision': round(precision, 2), 'Recall': round(recall, 2), 'F1-Score': round(f1, 2)}

# Save the results
logging.info("Saving results")
results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
results_df.to_csv('../data/results.csv', index=True)
logging.info("Results saved successfully!")