### Q2

# Import libraries
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import re

# Load data
ds = pd.read_csv("_ECON424_F2025_FINAL_PC_WAGE_SIMULATION_DATA.csv")

# Extract names
def extract_full_name(text):
    # Matches: CapitalizedWord CapitalizedWord
    match = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", str(text))
    if match:
        return match.group(1), match.group(2)  # first, last
    return None, None

ds["first_name"], ds["last_name"] = zip(*ds["discussion_between_two_people"].apply(extract_full_name))
print(ds.head())

# Separate wage into hired/not hired
ds_hired = ds[ds["wage"] != "not hired"].copy()
ds_not_hired = ds[ds["wage"] == "not hired"].copy()

print(ds_hired.head())

print(ds_not_hired.head())

# Convert wage from string to numeric
ds_hired["wage"] = pd.to_numeric(ds_hired["wage"], errors="coerce")

# Compute quartiles of wages
q1 = ds_hired["wage"].quantile(0.25)
median = ds_hired["wage"].quantile(0.50)
q3 = ds_hired["wage"].quantile(0.75)

ds_hired["wage_quartile"] = pd.qcut(ds_hired["wage"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

# Wage distribution
plt.figure(figsize=(10, 6))

sns.histplot(ds_hired["wage"], bins=30, kde=False, color="skyblue")

plt.axvline(q1, color="red", linestyle="--", linewidth=2, label=f"Q1 = {q1:.0f}")
plt.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median = {median:.0f}")
plt.axvline(q3, color="blue", linestyle="--", linewidth=2, label=f"Q3 = {q3:.0f}")

plt.title("Histogram of Wages with Quartile Markers")
plt.xlabel("Wage")
plt.ylabel("Frequency")
plt.legend()

plt.show()

# Heatmap
pivot = pd.crosstab(ds_hired["first_name"], ds_hired["wage_quartile"])
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

plt.figure(figsize=(14, 30))
sns.heatmap(pivot_pct, cmap="YlGnBu", linewidths=0.3)

plt.title("Name Frequency Across Wage Quartiles")
plt.xlabel("Wage Quartile")
plt.ylabel("First Name")

plt.show()

# First name hired vs. not hired heatmap
hired_counts = ds_hired["first_name"].value_counts()
not_hired_counts = ds_not_hired["first_name"].value_counts()
pivot_hire = pd.DataFrame({
    "Not Hired": not_hired_counts,
    "Hired": hired_counts
}).fillna(0)
pivot_hire_pct = pivot_hire.div(pivot_hire.sum(axis=1), axis=0)

plt.figure(figsize=(14, 30))

sns.heatmap(
    pivot_hire_pct,
    cmap="YlGnBu",
    linewidths=0.3,
    annot=False
)

plt.title("Hiring Outcome Percentages by First Name")
plt.xlabel("Outcome")
plt.ylabel("First Name")
plt.show()

# First name hired vs. not hired heatmap
hired_counts = ds_hired["last_name"].value_counts()
not_hired_counts = ds_not_hired["last_name"].value_counts()
pivot_hire = pd.DataFrame({
    "Not Hired": not_hired_counts,
    "Hired": hired_counts
}).fillna(0)
pivot_hire_pct = pivot_hire.div(pivot_hire.sum(axis=1), axis=0)

plt.figure(figsize=(14, 40))

sns.heatmap(
    pivot_hire_pct,
    cmap="YlGnBu",
    linewidths=0.3,
    annot=False
)

plt.title("Hiring Outcome Percentages by Last Name")
plt.xlabel("Outcome")
plt.ylabel("Last Name")
plt.show()

ds["hired"] = (ds["wage"] != "not hired").astype(int)

ds["text"] = (
    ds["discussion_between_two_people"].fillna("") +
    ds["chatbot_summary"].fillna("") +
    ds["chatbot_interview"].fillna("")
)

def clean_text(text):
    text = str(text).lower()
    
    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove tokens like "64", "67", "1200"
    text = re.sub(r"\b[0-9]+\b", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

ds["text"] = ds["text"].apply(clean_text)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)

X_text = tfidf.fit_transform(ds["text"])
y = ds["hired"]

logreg = LogisticRegression(max_iter=500)
logreg.fit(X_text, y)

import numpy as np
import pandas as pd

feature_names = tfidf.get_feature_names_out()
coef = logreg.coef_[0]

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "coef": coef
})

top_pos = feat_imp.sort_values("coef", ascending=False).head(20)
top_neg = feat_imp.sort_values("coef").head(20)

# Features that increase probability of getting hired
plt.figure(figsize=(10, 6))
sns.barplot(x=top_pos["coef"], y=top_pos["feature"], color="green")
plt.title("Top Words/Phrases Predicting Hiring")
plt.xlabel("Coefficient (Importance)")
plt.ylabel("Word / Phrase")
plt.show()

# Features that decrease probability of getting hired
plt.figure(figsize=(10, 6))
sns.barplot(x=top_neg["coef"], y=top_neg["feature"], color="red")
plt.title("Top Words/Phrases Predicting NOT Being Hired")
plt.xlabel("Coefficient (Importance)")
plt.ylabel("Word / Phrase")
plt.show()