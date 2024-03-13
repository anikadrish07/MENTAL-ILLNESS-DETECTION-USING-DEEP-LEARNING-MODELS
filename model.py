import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.dropna(how='any', inplace=True)
    df['text'] = df['Title'] + ': ' + df['Text']
    df['target'] = df['Subreddit'].copy()
    df.drop(['Text', 'Title', 'Subreddit'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    return train, val, test

def vectorize_text(train_text, val_text, test_text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    mod = vectorizer.fit(train_text)
    X_train = mod.transform(train_text)
    X_val = mod.transform(val_text)
    X_test = mod.transform(test_text)
    return X_train, X_val, X_test, mod

def encode_labels(train_target, val_target, test_target):
    le = LabelEncoder()
    mod2 = le.fit(train_target)
    y_train = mod2.transform(train_target)
    y_val = mod2.transform(val_target)
    y_test = mod2.transform(test_target)
    return y_train, y_val, y_test, mod2

def train_model(X_train, y_train):
    clf = LogisticRegression(random_state=0, max_iter=1000, solver='saga')  # Try different solver
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

def print_prediction(txt, vectorizer, clf, label_encoder):
    txt_tfidf = vectorizer.transform([txt])[0]
    pred = clf.predict(txt_tfidf)
    return label_encoder.inverse_transform(pred)[0]


def main():
    # Load data
    df = load_data('Mental-health-related-subreddits.csv')
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data
    train, val, test = split_data(df)
    
    # Vectorize text
    X_train, X_val, X_test, vectorizer = vectorize_text(train['text'], val['text'], test['text'])
    
    # Encode labels
    y_train, y_val, y_test, label_encoder = encode_labels(train['target'], val['target'], test['target'])

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train model
    clf = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(clf, X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()
