from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from data_preparation import load_and_preprocess_data

def train_and_evaluate_naive_bayes():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Initialize and train the model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    
    print("Naive Bayes Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    train_and_evaluate_naive_bayes()