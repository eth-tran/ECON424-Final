### Q1

### a)

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
import lightgbm as lgb

# Load data
train_small = pd.read_csv("424_F2025_Final_PC_small_train_v1.csv")
train_large = pd.read_csv("424_F2025_Final_PC_large_train_v1.csv")
test_ds = pd.read_csv("424_F2025_Final_PC_test_without_response_v1.csv")
ds = pd.concat([train_small, train_large], ignore_index=True)

text_cols = ["headline", "pros", "cons"]
cat_cols = ["firm", "job_title"]
num_cols = ["year_review"]
target = "rating"

# Fill missing
for c in text_cols:
    ds[c] = ds[c].fillna("")
    test_ds[c] = test_ds[c].fillna("")
for c in cat_cols:
    ds[c] = ds[c].fillna("Unknown")
    test_ds[c] = test_ds[c].fillna("Unknown")

# Feature engineering
ds["full_text"] = ds["headline"] + " " + ds["pros"] + " " + ds["cons"]
test_ds["full_text"] = test_ds["headline"] + " " + test_ds["pros"] + " " + test_ds["cons"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    ds, ds[target], test_size=0.2, random_state=42
)

# Column transformer
tfidf = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 2),
    stop_words="english"
)

ohe = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("text", tfidf, "full_text"),
        ("cat", ohe, cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# Train model
pipeline.fit(X_train, y_train)

# Training accuracy
train_pred = pipeline.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training Accuracy:", train_acc * 100)

# Validation accuracy
val_pred = pipeline.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("Validation Accuracy:", val_acc * 100)

# Fit on full data
pipeline.fit(ds, ds[target])

# Predict
test_pred = pipeline.predict(test_ds)

# Save prediction CSV in competition format
out = pd.DataFrame({
    "prediction": test_pred
})
out.to_csv("Q1_submission_predictions.csv", index=False)
print("Saved predictions to Q1_submission_predictions.csv")

### b)

# Word cloud
def make_wordcloud(text, title):
    wc = WordCloud(width=800, height=500, background_color="white").generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.show()

text_1 = " ".join(ds[ds[target] == 1]["full_text"].tolist())
text_5 = " ".join(ds[ds[target] == 5]["full_text"].tolist())

make_wordcloud(text_1, "Word Cloud — 1-Star Reviews")
make_wordcloud(text_5, "Word Cloud — 5-Star Reviews")

# Feature importance

# Extract fitted components
fitted_prep = pipeline.named_steps["prep"]
fitted_model = pipeline.named_steps["model"]

# ---- 1. Text feature names (TF-IDF) ----
tfidf_fitted = fitted_prep.named_transformers_["text"]
tfidf_features = tfidf_fitted.get_feature_names_out()

# ---- 2. OneHotEncoder feature names ----
ohe_fitted = fitted_prep.named_transformers_["cat"]
ohe_features = ohe_fitted.get_feature_names_out(cat_cols)

# ---- 3. Numeric features ----
numeric_features = num_cols

# Combine all feature names in correct order
all_features = np.concatenate([
    tfidf_features,
    ohe_features,
    numeric_features
])

# ---- 4. Extract feature importances from LightGBM ----
importances = fitted_model.feature_importances_

# Safety check (dimension alignment)
print("Number of features:", len(all_features))
print("Importance values:", len(importances))

# Create DataFrame
fi = pd.DataFrame({
    "feature": all_features,
    "importance": importances
})

# Sort and keep top 30 (otherwise text features dominate graph size)
fi_top = fi.sort_values("importance", ascending=False).head(30)

# ---- 5. Plot Feature Importance ----
plt.figure(figsize=(10, 12))
sns.barplot(
    data=fi_top,
    x="importance",
    y="feature",
    palette="viridis"
)
plt.title("Top 30 Feature Importances — LightGBM")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature distribution for train vs. test data
for col in num_cols:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(ds[col], label="Train", fill=True)
    sns.kdeplot(test_ds[col], label="Test", fill=True)
    plt.title(f"Distribution of {col}: Train vs Test")
    plt.legend()
    plt.show()

for col in cat_cols:
    plt.figure(figsize=(8, 5))
    train_counts = ds[col].value_counts(normalize=True).head(20)
    test_counts = test_ds[col].value_counts(normalize=True).head(20)
    compare_df = pd.DataFrame({"Train": train_counts, "Test": test_counts}).fillna(0)

    compare_df.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Category Distribution: {col}")
    plt.show()

# Prediction-error distribution
errors = y_val - val_pred
plt.figure(figsize=(8, 4))
sns.histplot(errors, bins=20, kde=True)
plt.title("Prediction Error Distribution (y − ŷ)")
plt.show()

### c)

# Average rating by presence of common words
from sklearn.feature_extraction.text import CountVectorizer

# Use a small vocabulary of common managerial keywords
keywords = [
    "management", "pay", "salary", "benefits", "hours",
    "culture", "team", "workload", "training", "flexible",
    "stress", "opportunity", "growth", "support"
]

cv = CountVectorizer(vocabulary=keywords, binary=True)
X_keywords = cv.fit_transform(ds["full_text"])

keyword_df = pd.DataFrame(X_keywords.toarray(), columns=keywords)
keyword_df["rating"] = ds["rating"]

avg_rating = keyword_df.groupby(lambda x: x)[keywords].mean()
avg_rating = keyword_df.groupby("rating").mean()

plt.figure(figsize=(12,6))
sns.barplot(
    x=keywords,
    y=[ds[ds["full_text"].str.contains(w, case=False)].rating.mean() for w in keywords],
    palette="coolwarm"
)
plt.xticks(rotation=45)
plt.title("Average Employer Rating When Review Contains Each Keyword")
plt.ylabel("Average Rating")
plt.xlabel("Keyword Appearing in Review")
plt.tight_layout()
plt.show()

# Sentiment score vs. rating
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

ds["pros_sent"] = ds["pros"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
ds["cons_sent"] = ds["cons"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

sent_group = ds.groupby("rating")[["pros_sent", "cons_sent"]].mean().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(data=sent_group, x="rating", y="pros_sent", marker="o", label="Pros Sentiment")
sns.lineplot(data=sent_group, x="rating", y="cons_sent", marker="o", label="Cons Sentiment")
plt.title("Average Sentiment of Pros/Cons by Star Rating")
plt.ylabel("Sentiment Score")
plt.xlabel("Rating")
plt.grid(True)
plt.show()