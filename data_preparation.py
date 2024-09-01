import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    # Load the credit card fraud dataset
    # dataset = load_dataset("liberatoratif/Credit-card-fraud-detection")

    # # Convert to pandas DataFrame
    # df = pd.DataFrame(dataset['train'])
    df = pd.read_csv("Dataset/creditcard.csv")
    # Select features and target
    features = [col for col in df.columns if col != 'Class']
    target = 'Class'

    # Split the data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Positive class ratio in training set: {sum(y_train)/len(y_train):.2%}")
    print(f"Positive class ratio in test set: {sum(y_test)/len(y_test):.2%}")