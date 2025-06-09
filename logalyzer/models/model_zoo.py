from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout

def logistic_regression_model(**kwargs):
    '''
    This function creates a Logistic Regression model with the following parameters:
    - penalty: 'l1' or 'l2'
    - C: Regularization parameter
    - solver: 'liblinear' or 'saga'

    Parameters:
    penalty: str
        Regularization norm
    C: float
        Regularization parameter
    solver: str
        Optimization algorithm
    '''
    return LogisticRegression(**kwargs)

    
def random_forest_model(**kwargs):
    '''
    This function creates a Random Forest model with the following parameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of the tree
    - min_samples_split: Minimum number of samples required to split an internal node
    - min_samples_leaf: Minimum number of samples required to be at a leaf node

    Parameters:
    n_estimators: int
        Number of trees in the forest
    max_depth: int
        Maximum depth of the tree
    min_samples_split: int
        Minimum number of samples required to split an internal node
    min_samples_leaf: int
        Minimum number of samples required to be at a leaf node
    '''
    return RandomForestClassifier(**kwargs)

def svm_model(**kwargs):
    '''
    This function creates a Support Vector Machine model with the following parameters:
    - C: Regularization parameter
    - kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    - degree: Degree of the polynomial kernel function
    - gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'

    Parameters:
    C: float
        Regularization parameter
    kernel: str
        Kernel type
    degree: int
        Degree of the polynomial kernel function
    gamma: float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    '''
    return SVC(**kwargs)

def lstm_model(hidden_units, input_shape, output):
    '''
    This function creates a LSTM model with the following architecture:
    - Bidirectional LSTM layer with hidden_units units and return_sequences=True
    - Bidirectional LSTM layer with hidden_units units
    - A Dense Layer Network with Dropouts followed by Softmax Output.

    Parameters:
    hidden_units: int
        Number of hidden units in LSTM layers
    input_shape: tuple
        Shape of input data
    output: int
        Number of output classes
    '''
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(hidden_units))(x)
    x = Dense(64, activation='LeakyReLU')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='LeakyReLU')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
