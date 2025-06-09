from parsing.parse import HadoopLogParser
from preprocessing.preprocessing_functions import preprocess_log
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Parsing the Raw Log Files
input_dir = '../data/Hadoop/'
output_dir = '../data/parsed_data/'
log_format = '<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>'
depth = 5
similarity = 0.3
keep_para = False
output_file = 'merged_data.parquet'

if not os.path.exists(os.path.join(output_dir, output_file)):
    logging.info("=====> Dataset not parsed yet. Parsing the dataset")
    parser = HadoopLogParser(log_format=log_format, input_dir=input_dir, output_dir=output_dir, depth=depth, st=similarity, maxChild=250, keep_para=keep_para, output_file=output_file)

    for application_id in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, application_id)):
            continue

        if not os.path.isdir(os.path.join(output_dir, application_id)):
            os.makedirs(os.path.join(output_dir, application_id))
        log_files = [os.path.join(application_id, file) for file in os.listdir(os.path.join(input_dir, application_id)) if file.endswith('.log')]
        for log_file in log_files:
            parser.parse(log_file)

    # Merging the Parsed Logs into a Single CSV
    logging.info("=====> Merging the parsed logs")
    parser.merge_parsed_logs()
else:
    logging.info("=====> Dataset already parsed. Skipping the parsing step")

# Loading the Merged Data
logging.info("=====> Loading the merged data")
data = pd.read_parquet(os.path.join(output_dir, output_file))

logging.info("=====> Dropping unnecessary columns")
# Drop the 'LineId' column as it is not needed
data.drop(['LineId','Date', 'Time','Level', 'Process', 'Component', 'Content'], axis=1, inplace=True)

logging.info("=====> Printing the first few rows of the data")
# see the first few rows of the data
logging.info(data.head())

# Renaming the EventTemplate column to LogTemplate
data.rename(columns={'EventTemplate': 'LogTemplate'}, inplace=True)

# Preprocess the Event Templates as they will be used downstream
logging.info("=====> Preprocessing the Log Templates")
# Selecting Content column where it's not None
data = data[data['LogTemplate'].notna()]

data['LogTemplate'] = data['LogTemplate'].astype(str)
data['LogTemplate'] = data['LogTemplate'].apply(lambda x: preprocess_log(x))

logging.info("=====> Log Templates Preprocessed")

# Loading this data to a parquet file
logging.info("=====> Saving the preprocessed data")
data = data[['application_id', 'file_name', 'LogTemplate', 'EventId']]
data.to_parquet('../data/parsed_data/parsed_data.parquet')
logging.info("=====> Data saved to parsed_data.parquet")

