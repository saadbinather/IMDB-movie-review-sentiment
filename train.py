import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report






df = pd.read_csv("IMDB_Dataset.csv")

# 🔹 Define a text cleaning function
def clean_text(text):
    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Lowercase all text
    text = text.lower()

    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 5. Tokenize (split into words)
    words = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # 7. Join words back into a single string
    cleaned_text = ' '.join(words)

    return cleaned_text

# 🔹 Apply the cleaning function to all reviews
df['cleaned_review'] = df['review'].apply(clean_text)

# 🔹 Show sample
print(df[['review', 'cleaned_review']].head())


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# Option 2: TF-IDF (recommended)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_review'])


# 🔹 Check the shape (rows = reviews, columns = unique words)
print("TF-IDF shape:", X_tfidf.shape)

# 🔹 Example: convert to dense DataFrame if you want to inspect
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())





df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)

# 1)Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
# Predict first 5 reviews
sample_preds_lr = log_reg.predict(X_tfidf[:5])
print("\n🧠 Logistic Regression - First 5 Predictions:")
for i, p in enumerate(sample_preds_lr):
    print(f"{i+1}. Prediction: {'Positive' if p == 1 else 'Negative'}")
    print(f"   Review: {df['review'][i][:200]}...\n")
print("🔹 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


#2) Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
#Predict first 5 reviews
sample_preds_nb = nb.predict(X_tfidf[:5])
print("\n🧠 Naive Bayes - First 5 Predictions:")
for i, p in enumerate(sample_preds_nb):
    print(f"{i+1}. Prediction: {'Positive' if p == 1 else 'Negative'}")
    print(f"   Review: {df['review'][i][:200]}...\n")
print("🔹 Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


#3) Linear SVM
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

# 🔸 Predict first 5 reviews
sample_preds_svm = svm_clf.predict(X_tfidf[:5])
print("\n🧠 Linear SVM - First 5 Predictions:")
for i, p in enumerate(sample_preds_svm):
    print(f"{i+1}. Prediction: {'Positive' if p == 1 else 'Negative'}")
    print(f"   Review: {df['review'][i][:200]}...\n")
print("🔹 Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))



while True:
    # 🔹 USER INPUT TEST
    print("\n💬 Test your own review!")
    user_input = input("Enter a movie review: ")

    # Clean and vectorize input
    cleaned_input = clean_text(user_input)
    input_tfidf = tfidf_vectorizer.transform([cleaned_input])

    # Predict with all 3 models
    pred_lr = log_reg.predict(input_tfidf)[0]
    pred_nb = nb.predict(input_tfidf)[0]
    pred_svm = svm_clf.predict(input_tfidf)[0]

    print("\nModel Predictions:")
    print(f"  Logistic Regression  → {'Positive' if pred_lr == 1 else 'Negative'}")
    print(f"  Naive Bayes          → {'Positive' if pred_nb == 1 else 'Negative'}")
    print(f"  Linear SVM           → {'Positive' if pred_svm == 1 else 'Negative'}")













