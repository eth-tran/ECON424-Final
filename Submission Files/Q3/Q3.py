### Q3

# Import libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import re

# Load data
df = pd.read_csv("ECON424_F2025_FINAL_PC_STOCK_SIMULATION_DATA.csv")

# Extract year
def extract_year(text):
    header = str(text)[:60]
    match = re.search(r"20\d{2}", header)
    return int(match.group(0)) if match else np.nan

df["year"] = df["annual_report"].apply(extract_year)

# Extract company name
def clean_header(text):
    text = str(text).strip()
    text = text.replace("**", "")           # remove markdown bold
    text = text.replace("#", "")            # remove markdown headings
    text = re.sub(r"\(.*?\)", "", text)     # remove ticker symbols (AAPL)
    text = re.sub(r"proudly presents.*", "", text, flags=re.IGNORECASE)
    text = text.strip(" -–—")               # remove trailing punctuation
    return text

df["header_clean"] = df["annual_report"].apply(clean_header)

def extract_company(cleaned):
    # Pattern A: "Company ... Annual Report"
    m = re.match(r"^(.*?)(?:\s*\d{4})?\s*[-–—]?\s*Annual Report", cleaned, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if len(name) > 1:
            return name
    
    # Pattern B: "Company ... Annual ..."
    m = re.match(r"^(.*?)\s*[-–—]?\s*Annual", cleaned, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if len(name) > 1:
            return name
    
    # Pattern C: Stop before year
    m = re.match(r"^(.*?)\s*\d{4}", cleaned)
    if m:
        name = m.group(1).strip()
        if len(name) > 1:
            return name
    
    # Fallback: take first line up to 4 words
    name = " ".join(cleaned.split()[:4])
    return name

df["company"] = df["header_clean"].apply(extract_company)
df["company"] = df["company"].str.strip(" .,-–—")
df["company"] = df["company"].str.replace(r"\s+", " ", regex=True)
df["company"].value_counts().head(50)

# Aggregate stock price by year
annual_data = df.groupby("year")["stock_price"].agg(
    avg_price="mean",
    median_price="median",
    max_price="max",
    min_price="min"
).reset_index()

print(annual_data.head())

# Stock price over time
panel = (
    df.groupby(["company", "year"])["stock_price"]
      .median()
      .reset_index()
)

plt.figure(figsize=(14, 8))

for c in panel["company"].unique():
    subset = panel[panel["company"] == c]
    plt.plot(subset["year"], subset["stock_price"], alpha=0.4)

plt.title("Stock Price Trends Across All Companies")
plt.xlabel("Year")
plt.ylabel("Median Stock Price")
plt.grid(True)
plt.show()

# Extract keywords over time
keywords = ["revenue", "profit", "growth", "decline", "expenses", "investment", "risk"]

for word in keywords:
    df[word] = df["annual_report"].str.lower().str.count(word)

keyword_trends = df.groupby("year")[keywords].sum()

# Keyword trends
plt.figure(figsize=(12,6))
for word in keywords:
    plt.plot(keyword_trends.index, keyword_trends[word], label=word)

plt.legend()
plt.title("Frequency of Economic Terms in Annual Reports Over Time")
plt.xlabel("Year")
plt.ylabel("Keyword Frequency")
plt.grid(True)
plt.show()

# Summary
summary_table = annual_data.describe()
print(summary_table)