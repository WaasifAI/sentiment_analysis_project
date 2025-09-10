import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv(r"D:/PythonProject/final_dataset.csv")
print("Dataset sucessfully loaded")
print(df.head())

# Step 2: Clean the dataset
df = df.dropna(subset=['comment'])   # remove rows with empty comments

def clean_text(text):
    text = re.sub(r'\n', ' ', text)          # remove newline chars
    text = re.sub(r'[^a-zA-Z ]', '', text)   # keep only letters
    text = text.lower().strip()              # lowercase and strip spaces
    return text

df['clean_comment'] = df['comment'].apply(clean_text)

# Step 3: Convert stars into sentiment labels
def star_to_sentiment(star):
    if "5.0" in star or "4.0" in star:
        return "positive"
    elif "3.0" in star:
        return "neutral"
    else:
        return "negative"

df['sentiment'] = df['stars'].apply(star_to_sentiment)

print("Sentiment distribution:")
print(df['sentiment'].value_counts())

# Step 4: TF-IDF vectorization
X = df['clean_comment']
y = df['sentiment']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Step 5: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Step 6: Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict sentiment for all rows
df['predicted_sentiment'] = model.predict(X_tfidf)

# Step 9: Compare brands
brand_sentiment = (
    df.groupby('Brand')['predicted_sentiment']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0) * 100
)

print("Brand-wise Sentiment Distribution (%):")
print(brand_sentiment)

# Plot
brand_sentiment[['positive', 'neutral', 'negative']].plot(
    kind='bar', stacked=True, figsize=(10,6), colormap='viridis'
)

plt.title("Brand-wise Sentiment Analysis (%)")
plt.ylabel("Percentage")
plt.xlabel("Brand")
plt.legend(title="Sentiment")
plt.show()



