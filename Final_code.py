import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN

# Load the dataset
file_path = "F:\CINDY(copy)\VIT MTECH INT BA\SEM4\CSE3040-EDA\Destination Reviews (final).csv"
df = pd.read_csv(file_path)

# Display basic info
df.info()
print(df.head())

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Summary statistics
print(df.describe(include='all'))

# Plot the distribution of destinations
plt.figure(figsize=(12, 6))
sns.countplot(y=df['Destination'], order=df['Destination'].value_counts().index[:20])
plt.title("Top 20 Most Reviewed Destinations")
plt.show()

# Word Cloud for Reviews
text = " ".join(df['Review'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Reviews")
plt.show()

# Association Analysis: Find frequently visited together places
grouped_data = df.groupby(['District', 'Timespan'])['Destination'].apply(list).reset_index()
transactions = grouped_data["Destination"].tolist()
pair_counts = Counter()
for destinations in transactions:
    for pair in combinations(sorted(destinations), 2):
        pair_counts[pair] += 1
pair_df = pd.DataFrame(pair_counts.items(), columns=["Destination Pair", "Count"])
pair_df = pair_df.sort_values(by="Count", ascending=False)

# Print associative rule mining results
print("Top 10 Most Frequently Visited Destination Pairs:\n")
print(pair_df.head(10).to_string(index=False))

# DBSCAN Clustering on Reviews
df_sample = df.sample(n=2000, random_state=42)  # Reduce dataset size
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_sample = vectorizer.fit_transform(df_sample["Review"].dropna())
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X_sample)
dbscan = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
clusters_sample = dbscan.fit_predict(X_reduced)
df_sample["Cluster"] = clusters_sample

# Print DBSCAN clustering results
print("\nDBSCAN Clustering Results (Sample 10 Reviews):\n")
print(df_sample[["Destination", "District", "Review", "Cluster"]].head(10).to_string(index=False))

# Print the number of reviews in each cluster
print("\nCluster Distribution:\n")
print(df_sample["Cluster"].value_counts())


# ---------------------------
# ✨ Predictive Modeling Phase ✨
# ---------------------------

from sklearn.feature_selection import VarianceThreshold
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Sentiment as a label (feature engineering)
df['Polarity'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Sentiment_Label'] = df['Polarity'].apply(lambda x: 1 if x > 0 else 0)

# Step 2: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['Review'])

# Step 3: Feature Selection (Variance Threshold)
selector = VarianceThreshold(threshold=0.001)
X_selected = selector.fit_transform(X)
y = df['Sentiment_Label']

# Step 4: Handle Imbalance (if any) using upsampling
df_features = pd.DataFrame(X_selected.toarray())
df_features['label'] = y.values

df_majority = df_features[df_features.label == 0]
df_minority = df_features[df_features.label == 1]

if len(df_majority) > len(df_minority):
    df_minority_upsampled = resample(df_minority, replace=True,
                                     n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
else:
    df_majority_upsampled = resample(df_majority, replace=True,
                                     n_samples=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_upsampled, df_minority])

X_balanced = df_balanced.drop('label', axis=1)
y_balanced = df_balanced['label']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 6: Apply Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Naive Bayes': MultinomialNB()
}

print('\n🔍 Predictive Model Comparison (Balanced Sentiment Prediction):')
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name} Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))
