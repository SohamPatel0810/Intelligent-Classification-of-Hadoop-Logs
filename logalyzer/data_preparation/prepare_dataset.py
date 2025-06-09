import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import logging

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# For Approach 1: TF-IDF on Event IDs
from sklearn.feature_extraction.text import TfidfVectorizer

# For Approach 2: BERT Encoding on Event Messages using sentence transformers
from sentence_transformers import SentenceTransformer

# Helper Functions
def _load_parsed_data(output_dir, output_file):
    """Load the merged parsed data from a parquet file."""
    logging.info(f"Loading parsed data from {output_dir}/{output_file}")
    data = pd.read_parquet(os.path.join(output_dir, output_file))
    logging.info("Parsed data loaded successfully")
    return data

# Create windows of log messages
def _create_overlapping_windows(group, window_size, overlap):
    step_size = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(group) - window_size + 1, step_size):
        window = group.iloc[start:start + window_size]
        joined_content = ' '.join(window['LogTemplate'].tolist())
        joined_event_ids = ','.join(window['EventId'].tolist())
        windows.append((window['application_id'].iloc[0], window['file_name'].iloc[0], joined_content, joined_event_ids))
    return windows

def _create_tf_idf_matrix(data, event_id_column, max_features=1000):
    """Create a TF-IDF matrix from the event messages."""
    logging.info(f"Creating TF-IDF matrix with max_features={max_features}")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[event_id_column])
    logging.info("TF-IDF matrix created successfully")
    return tfidf_matrix, tfidf_vectorizer

def _create_log_window_embeddings(data, event_message_column):
    """Create SBERT embeddings for the event messages. Use the all-MiniLM-L6-v2 model."""
    logging.info("Creating SBERT embeddings for the event messages")
    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Tokenize the event messages
    data_to_encode = data[event_message_column].tolist()
    # Sentence Embeddings
    data_embeddings = model.encode(data_to_encode, convert_to_tensor=False)
    logging.info("BERT embeddings created successfully")
    return data_embeddings

def _save_tfidf_tokenizer(tfidf_vectorizer, output_dir, output_file):
    """Save the TF-IDF vectorizer to a file."""
    logging.info(f"Saving TF-IDF vectorizer to {output_dir}/{output_file}")
    with open(os.path.join(output_dir, output_file), 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    logging.info("TF-IDF vectorizer saved successfully")

def prepare_log_dataset(output_dir, output_file, window_size, window_overlap, label_file_path, event_id_column='EventIds', event_message_column='LogWindows', max_features=1000, dataset_output_dir='.'):
    """
    Prepare the log dataset for machine learning.
    Args:
        output_dir (str): The directory path where the parsed log data is stored.
        output_file (str): The name of the parsed log data file.
        window_size (int): The size of the sliding window for creating overlapping log message windows.
        window_overlap (int): The overlap between consecutive log message windows.
        label_file_path (str): The file path of the label file.
        event_id_column (str, optional): The name of the column containing event IDs. Defaults to 'EventIds'.
        event_message_column (str, optional): The name of the column containing log messages. Defaults to 'LogWindows'.
        max_features (int, optional): The maximum number of features to consider for TF-IDF. Defaults to 1000.
        dataset_output_dir (str, optional): The directory path to save the prepared dataset. Defaults to '.'.
    Returns:
        tuple: A tuple containing the training and testing data along with their corresponding labels.
    Raises:
        FileNotFoundError: If the parsed log data file or label file is not found.
    """

    logging.info("Starting dataset preparation")

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        logging.error(f"Directory not found: {output_dir}")
        raise FileNotFoundError(f"Directory not found: {output_dir}")
    
    # Check if dataset output directory exists. If not, create it.
    if not os.path.exists(dataset_output_dir):
        logging.info(f"Creating dataset output directory: {dataset_output_dir}")
        os.makedirs(dataset_output_dir)

    # Load the parsed data
    data = _load_parsed_data(output_dir, output_file)
    
    # Group the data by application_id and file_name
    logging.info("Grouping data by application_id and file_name")
    grouped_data = data.groupby(['application_id', 'file_name'])

    # Create overlapping windows of log messages
    logging.info("Creating overlapping windows of log messages")
    windows = []
    for _, group in grouped_data:
        windows.extend(_create_overlapping_windows(group, window_size, window_overlap))
    window_data = pd.DataFrame(windows, columns=['application_id', 'file_name', event_message_column, event_id_column])

    logging.info("Loading labels from label file")
    labels_df = pd.read_csv(label_file_path)
    final_df = window_data.merge(labels_df, left_on='application_id', right_on='application_id', how='left') 
    
    # Drop rows with NaN labels if any
    final_df = final_df.dropna(subset=['label'])
    labels = final_df['label'].values
    
    # Create the TF-IDF matrix
    tfidf_matrix, tfidf_vectorizer = _create_tf_idf_matrix(final_df, event_id_column, max_features)
    
    # Create the BERT embeddings
    data_embeddings = _create_log_window_embeddings(final_df, event_message_column)

    # Save the TF-IDF vectorizer
    _save_tfidf_tokenizer(tfidf_vectorizer, dataset_output_dir, 'tfidf_vectorizer.pkl')
    
    # Merge the TF-IDF matrix and SBERT embeddings
    logging.info("Merging TF-IDF matrix and BERT embeddings")
    merged_data = np.concatenate([tfidf_matrix.toarray(), data_embeddings], axis=1)

    # Train test split
    logging.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(merged_data, labels, test_size=0.3, random_state=42)

    # Save the dataset as parquet files
    logging.info("Saving training data to parquet file")
    train_data = pd.DataFrame(data=X_train)
    train_data['Label'] = y_train
    train_data.to_parquet(os.path.join(dataset_output_dir, 'train_data.parquet'))

    logging.info("Saving testing data to parquet file")
    test_data = pd.DataFrame(data=X_test)
    test_data['Label'] = y_test
    test_data.to_parquet(os.path.join(dataset_output_dir, 'test_data.parquet'))

    logging.info("Dataset preparation complete")
    return X_train, X_test, y_train, y_test