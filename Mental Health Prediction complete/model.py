import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def downsample_data(df, target_count=100000, target_label='depression', random_state=42):
    # Filter the DataFrame to only include instances with the target label
    target_data = df[df['target'] == target_label]
    
    # Check the count of target data
    target_count_current = target_data.shape[0]
    
    # If the target data count is already less than or equal to the target count, no downsampling is needed
    if target_count_current <= target_count:
        print(f"{target_label.capitalize()} data count is already less than or equal to the target count.")
        return target_data
    else:
        # Downsample the target data to the target count
        downsampled_target_data = target_data.sample(n=target_count, random_state=random_state)
        print(f"Downsampled {target_label} data count:", downsampled_target_data.shape[0])
        return downsampled_target_data

def split_data(df, test_size=0.2, random_state=0):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train, test_size=test_size, random_state=random_state)
    return train, val, test

def vectorize_text(train_text, val_text, test_text, max_features=5000):
    corpus = train_text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    mod = vectorizer.fit(corpus)
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

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, random_state=0):
    clf = LogisticRegression(random_state=random_state, max_iter=1000).fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    train_score = accuracy_score(y_train, y_train_pred)
    val_score = accuracy_score(y_val, y_val_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    return clf, train_score, val_score, test_score

def print_score(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

def print_prediction(txt, clf, vectorizer, label_encoder):
    txt_tfidf = vectorizer.transform([txt])
    pred = clf.predict(txt_tfidf)
    prediction = label_encoder.inverse_transform(pred)
    return prediction
