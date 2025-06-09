import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def _to_lower(text):
    return text.lower()

def _remove_numbers(text):
    return re.sub(r'\d+', '', text)

def _remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

def _remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text)

def _remove_line_breaks(text):
    return text.replace('\n', ' ')

def _remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess_log(log):
    log = _to_lower(log)
    log = _remove_numbers(log)
    log = _remove_special_characters(log)
    log = _remove_extra_whitespaces(log)
    log = _remove_line_breaks(log)
    log = _remove_stopwords(log)
    return log


