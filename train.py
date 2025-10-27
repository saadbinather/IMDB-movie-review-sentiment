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
print(X_tfidf_df.head())




df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)

#1 Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("\n🔹 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))




#2 Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("\n🔹 Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))




#3 Linear SVM
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("\n🔹 Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
