import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def load_data(path="data/expenses.csv"):
    df = pd.read_csv(path)
    return df
def split_features(df):
    X_text = df["description"]
    X_amount = df["amount"]
    y = df["category"]
    return X_text, X_amount, y
def vectorize_text(X_text):
    vectorizer = TfidfVectorizer()
    X_text_vec = vectorizer.fit_transform(X_text)
    return X_text_vec, vectorizer
def combine_features(X_text_vec, X_amount):
    X_amount = X_amount.values.reshape(-1, 1)
    X_final = hstack((X_text_vec, X_amount))
    return X_final
def preprocess_data():
    df = load_data()
    X_text, X_amount, y = split_features(df)
    X_text_vec, vectorizer = vectorize_text(X_text)
    X_final = combine_features(X_text_vec, X_amount)
    return X_final, y, vectorizer
if __name__ == "__main__":
    X, y, vectorizer = preprocess_data()
    print("Features shape:", X.shape)
    print("Labels count:", len(y))
