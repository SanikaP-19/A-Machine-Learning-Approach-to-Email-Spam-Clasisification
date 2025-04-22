import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def analyze_dataset(file_path, algorithm):
    try:
        # Load the dataset with better error handling
        data = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')
        # Load the dataset

# Rename the columns to match the expected ones
        data.columns = ['Email Content', 'Spam/Ham']  # Adjust column names accordingly

        # Check required columns
        if 'Email Content' not in data.columns or 'Spam/Ham' not in data.columns:
            return {'error': 'Dataset must contain "Email Content" and "Spam/Ham" columns'}
        
        # Drop rows with missing values
        data = data.dropna(subset=['Email Content', 'Spam/Ham'])
        if data.empty:
            return {'error': 'Dataset contains missing values. Please clean the data and try again'}
        
        # Preprocess target column
        X = data['Email Content']
        y = data['Spam/Ham'].apply(lambda x: 1 if x.lower() == 'spam' else 0)
        
        # Feature extraction
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(X)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the model
        model = None
        if algorithm == 'Linear Regression':
            model = LogisticRegression()
        elif algorithm == 'SVM':
            model = SVC()
        elif algorithm == 'KNN':
            model = KNeighborsClassifier()
        elif algorithm == 'Naive Bayes':
            model = MultinomialNB()
        else:
            return {'error': 'Invalid Algorithm selected'}
        
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        #Generate confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        
        #Save confusion matrix as an image
        cm_path = f"static/confusion_matrix_{algorithm.replace(' ', '_')}.png"
        plt.figure(figsize=(8,6))
        sns.heatmap(cm,annot = True,fmt='d',cmap='Blues',xticklabels=['Ham','Spam'],yticklabels=['Ham','Spam'])
        plt.title(f"{algorithm}-Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(cm_path)
        plt.close()
        
        return {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'details': report,
            'confusion_matrix_path': cm_path
        }
    
    except Exception as e:
        return {'error': f'Failed to analyze the dataset: {str(e)}'}
