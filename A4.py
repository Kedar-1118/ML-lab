import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
from nltk.tokenize import word_tokenize


# -------------------------------
# PDF Text Extraction
# -------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# -------------------------------
# N-Gram Extraction
# -------------------------------
def extractNgrams(my_text):
    tokens = word_tokenize(my_text)

    unigrams_dict, bigrams_dict, trigrams_dict, quadgrams_dict = {}, {}, {}, {}

    def cleanText(tokenList):
        return [token.lower() for token in tokenList if token.isalnum()]

    tokens = cleanText(tokens)

    for i in range(len(tokens) - 3):
        token_1, token_2, token_3, token_4 = tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]

        unigram = token_1
        bigram = f"{token_1} {token_2}"
        trigram = f"{bigram} {token_3}"
        quadgram = f"{trigram} {token_4}"

        unigrams_dict[unigram] = unigrams_dict.get(unigram, 0) + 1
        bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) + 1
        trigrams_dict[trigram] = trigrams_dict.get(trigram, 0) + 1
        quadgrams_dict[quadgram] = quadgrams_dict.get(quadgram, 0) + 1

    # Remaining unigrams
    for token in tokens[len(tokens) - 3:]:
        unigrams_dict[token] = unigrams_dict.get(token, 0) + 1

    # Remaining bigrams
    for i in range(len(tokens) - 3, len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) + 1

    # Last trigram
    trigram = f"{tokens[-3]} {tokens[-2]} {tokens[-1]}"
    trigrams_dict[trigram] = trigrams_dict.get(trigram, 0) + 1

    return [[unigrams_dict, bigrams_dict, trigrams_dict, quadgrams_dict]]


# -------------------------------
# Outlier Detection Functions
# -------------------------------
def robust_zscore(data):
    median = data.median()
    iqrange = data.quantile(0.75) - data.quantile(0.25)
    return (data - median) / iqrange if iqrange else 0


def zscore(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std if std else 0


def calculateMahalanobis(df):
    numeric_df = df[["frequency", "num_pdfs"]]

    mean_vec = np.mean(numeric_df, axis=0)
    cov_matrix = np.cov(numeric_df, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    def mahalanobis(x, mean, inv_cov):
        diff = x - mean
        return np.sqrt(diff.T @ inv_cov @ diff)

    df["mahalanobis_distance"] = numeric_df.apply(
        lambda row: mahalanobis(row.values, mean_vec, inv_cov_matrix), axis=1
    )
    return df["mahalanobis_distance"]


def MAD(df):
    data = np.array(df["frequency"])
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    z = (data - median) / mad
    return mad


# -------------------------------
# Load PDFs and Extract Text
# -------------------------------
pdf_folder = "batchspdf/"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

pdf_texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# Extract N-Grams
ngrams_list = []
for pdf_text in pdf_texts:
    ngrams_list.extend(extractNgrams(pdf_text))


# -------------------------------
# Create DataFrames for N-Grams
# -------------------------------
unigrams_df = pd.DataFrame(
    [unigram for ngram in ngrams_list for unigram in ngram[0].items()],
    columns=["unigram", "frequency"],
)
unigrams_df["num_pdfs"] = unigrams_df["unigram"].apply(
    lambda x: sum(1 for ngram in ngrams_list if x in ngram[0])
)
unigrams_df["frequency_per_pdf"] = unigrams_df["frequency"] / unigrams_df["num_pdfs"]

bigrams_df = pd.DataFrame(
    [bigram for ngram in ngrams_list for bigram in ngram[1].items()],
    columns=["bigram", "frequency"],
)
bigrams_df["num_pdfs"] = bigrams_df["bigram"].apply(
    lambda x: sum(1 for ngram in ngrams_list if x in ngram[1])
)
bigrams_df["frequency_per_pdf"] = bigrams_df["frequency"] / bigrams_df["num_pdfs"]

trigrams_df = pd.DataFrame(
    [trigram for ngram in ngrams_list for trigram in ngram[2].items()],
    columns=["trigram", "frequency"],
)
trigrams_df["num_pdfs"] = trigrams_df["trigram"].apply(
    lambda x: sum(1 for ngram in ngrams_list if x in ngram[2])
)
trigrams_df["frequency_per_pdf"] = trigrams_df["frequency"] / trigrams_df["num_pdfs"]

quadgrams_df = pd.DataFrame(
    [quadgram for ngram in ngrams_list for quadgram in ngram[3].items()],
    columns=["quadgram", "frequency"],
)
quadgrams_df["num_pdfs"] = quadgrams_df["quadgram"].apply(
    lambda x: sum(1 for ngram in ngrams_list if x in ngram[3])
)
quadgrams_df["frequency_per_pdf"] = quadgrams_df["frequency"] / quadgrams_df["num_pdfs"]


# -------------------------------
# Unigrams Analysis
# -------------------------------
unigrams_df["z_score"] = zscore(unigrams_df["frequency"])
unigrams_df["robust_z_score"] = robust_zscore(unigrams_df["frequency"])
mahalanobis_distance = calculateMahalanobis(unigrams_df)

plt.figure(figsize=(12, 6))
sns.histplot(unigrams_df["frequency"], bins=100)
plt.title("Unigrams Frequency Distribution")
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.show()

unigrams_std_z_score = unigrams_df["z_score"].std()
left_threshold_z_score = -2.5 * unigrams_std_z_score
right_threshold_z_score = 2.5 * unigrams_std_z_score

plt.figure(figsize=(12, 6))
sns.kdeplot(unigrams_df["z_score"], color="blue")
plt.axvline(x=0, color="r", linestyle="--")
plt.axvline(x=right_threshold_z_score, color="g", linestyle="--")
plt.axvline(x=left_threshold_z_score, color="g", linestyle="--")
plt.title("Unigrams Z-Score Distribution")
plt.xlabel("Z-Score")
plt.ylabel("Frequency")
plt.show()

unigrams_outliers_zscore = unigrams_df[
    (unigrams_df["z_score"] > right_threshold_z_score)
    | (unigrams_df["z_score"] < left_threshold_z_score)
]
unigrams_outliers_zscore["unigram"].to_csv("unigrams_outliers_zscore.csv", index=False)

robust_iqr = unigrams_df["robust_z_score"].quantile(0.75) - unigrams_df["robust_z_score"].quantile(0.25)
right_threshold = unigrams_df["robust_z_score"].quantile(0.75) + 1.5 * robust_iqr
left_threshold = unigrams_df["robust_z_score"].quantile(0.25) - 1.5 * robust_iqr

plt.figure(figsize=(12, 6))
sns.kdeplot(unigrams_df["robust_z_score"], color="blue")
plt.axvline(x=0, color="r", linestyle="--")
plt.axvline(x=right_threshold, color="g", linestyle="--")
plt.axvline(x=left_threshold, color="g", linestyle="--")
plt.title("Unigrams Robust Z-Score Distribution")
plt.xlabel("Robust Z-Score")
plt.ylabel("Frequency")
plt.show()

unigrams_outliers_robustzscore = unigrams_df[
    (unigrams_df["robust_z_score"] > right_threshold)
    | (unigrams_df["robust_z_score"] < left_threshold)
]
unigrams_outliers_robustzscore["unigram"].to_csv("unigrams_outliers_robustzscore.csv", index=False)

plt.figure(figsize=(12, 6))
plt.scatter(unigrams_df["frequency"], unigrams_df["num_pdfs"])
plt.title("Unigrams Frequency vs Number of PDFs")
plt.xlabel("Frequency")
plt.ylabel("Number of PDFs")
plt.show()

plt.figure(figsize=(24, 6))
sns.kdeplot(mahalanobis_distance, color="blue")
plt.axvline(x=2.0, color="g", linestyle="--")
plt.title("Unigrams Mahalanobis Distance Distribution")
plt.xlabel("Mahalanobis Distance")
plt.ylabel("Density")
plt.show()

unigrams_outliers_mahalanobis = unigrams_df[mahalanobis_distance > 2.0]
unigrams_outliers_mahalanobis["unigram"].to_csv("unigrams_outliers_mahalanobis.csv", index=False)


# -------------------------------
# Bigrams Analysis
# -------------------------------
bigrams_df["z_score"] = zscore(bigrams_df["frequency"])
bigrams_df["robust_z_score"] = robust_zscore(bigrams_df["frequency"])
bigrams_df["mahalanobis_distance"] = calculateMahalanobis(bigrams_df)

plt.figure(figsize=(24, 6))
sns.kdeplot(bigrams_df["z_score"], color="blue")
plt.axvline(x=0, color="r", linestyle="--")
plt.axvline(x=2.5, color="g", linestyle="--")
plt.axvline(x=-2.5, color="g", linestyle="--")
plt.xlabel("Z-Score")
plt.ylabel("Density")
plt.show()

bigrams_outlier_zscore = bigrams_df[
    (bigrams_df["z_score"] > 2.5) | (bigrams_df["z_score"] < -2.5)
]
bigrams_outlier_zscore["bigram"].to_csv("bigrams_outlier_zscore.csv", index=False)

iqr = bigrams_df["robust_z_score"].quantile(0.75) - bigrams_df["robust_z_score"].quantile(0.25)
a = bigrams_df["robust_z_score"].quantile(0.75) + 1.5 * iqr
b = bigrams_df["robust_z_score"].quantile(0.25) - 1.5 * iqr

plt.figure(figsize=(24, 6))
sns.kdeplot(bigrams_df["robust_z_score"], color="blue")
plt.axvline(x=0, color="r", linestyle="--")
plt.axvline(x=a, color="g", linestyle="--")
plt.axvline(x=b, color="g", linestyle="--")
plt.xlabel("Robust Z-Score")
plt.ylabel("Density")
plt.show()

bigrams_outlier_robust_zscore = bigrams_df[
    (bigrams_df["robust_z_score"] < b) | (bigrams_df["robust_z_score"] > a)
]
bigrams_outlier_robust_zscore["bigram"].to_csv("bigrams_outlier_robust_zscore.csv", index=False)

plt.figure(figsize=(24, 6))
sns.kdeplot(bigrams_df["mahalanobis_distance"], color="blue")
plt.axvline(x=1.4, color="r", linestyle="--")
plt.xlabel("Mahalanobis Distance")
plt.ylabel("Density")
plt.show()

bigrams_outlier_mahalanobis = bigrams_df[bigrams_df["mahalanobis_distance"] > 1.4]
bigrams_outlier_mahalanobis["bigram"].to_csv("bigrams_outlier_mahalanobis.csv", index=False)


# -------------------------------
# Trigrams Analysis
# -------------------------------
trigrams_df["z_score"] = zscore(trigrams_df["frequency"])
trigrams_df["robust_z_score"] = robust_zscore(trigrams_df["frequency"])
trigrams_df["mahalanobis_distance"] = calculateMahalanobis(trigrams_df)

plt.figure(figsize=(24, 6))
sns.kdeplot(trigrams_df["z_score"], color="blue")
plt.axvline(x= -2.5, color="r", linestyle="--")
plt.axvline(x= 2.5, color="g", linestyle="--")
plt.xlabel("Frequency")
plt.ylabel("Density")
plt.title("Trigrams Frequency Distribution")
plt.show()

trigrams_outlier_zscore = trigrams_df[
    (trigrams_df["z_score"] > 2.5) | (trigrams_df["z_score"] < -2.5)
]
trigrams_outlier_zscore["trigram"].to_csv("trigrams_outlier_zscore.csv", index=False)


left_threshold = trigrams_df["robust_z_score"].quantile(0.25) - 1.5 * (trigrams_df["robust_z_score"].quantile(0.75) - trigrams_df["robust_z_score"].quantile(0.25))
right_threshold = trigrams_df["robust_z_score"].quantile(0.75) + 1.5 * (trigrams_df["robust_z_score"].quantile(0.75) - trigrams_df["robust_z_score"].quantile(0.25))
plt.figure(figsize=(24, 6))
sns.kdeplot(trigrams_df["robust_z_score"], color="blue")
plt.axvline(x= left_threshold, color="r", linestyle="--")
plt.axvline(x= right_threshold, color="g", linestyle="--")
plt.xlabel("Frequency")
plt.ylabel("Density")
plt.title("Trigrams Frequency Distribution")
plt.show()

trigrams_outlier_robust_zscore = trigrams_df[
    (trigrams_df["robust_z_score"] > right_threshold) | (trigrams_df["robust_z_score"] < left_threshold)
]
trigrams_outlier_robust_zscore["trigram"].to_csv("trigrams_outlier_robust_zscore.csv", index=False)

plt.figure(figsize=(24, 6))
sns.kdeplot(trigrams_df["mahalanobis_distance"], color="blue")
plt.axvline(x=1.0, color="g", linestyle="--")
plt.xlabel("Mahalanobis Distance")
plt.ylabel("Density")
plt.title("Trigrams Mahalanobis Distance Distribution")
plt.show()

trigrams_outlier_mahalanobis = trigrams_df[trigrams_df["mahalanobis_distance"] > 1.0]
trigrams_outlier_mahalanobis["trigram"].to_csv("trigrams_outlier_mahalanobis.csv", index=False)

# -------------------------------
# Quadgrams Analysis
# -------------------------------
quadgrams_df["z_score"] = zscore(quadgrams_df["frequency"])
quadgrams_df["robust_z_score"] = robust_zscore(quadgrams_df["frequency"])
quadgrams_df["mahalanobis_distance"] = calculateMahalanobis(quadgrams_df)

plt.figure(figsize=(24, 6))
sns.kdeplot(quadgrams_df["z_score"], color="blue")
plt.axvline(x=0, color="r", linestyle="--")
plt.axvline(x=2.5, color="g", linestyle="--")
plt.axvline(x=-2.5, color="g", linestyle="--")
plt.xlabel("Z-Score")
plt.ylabel("Density")
plt.show()

quadgrams_outlier_zscore = trigrams_df[
    (quadgrams_df["z_score"] > 2.5) | (quadgrams_df["z_score"] < -2.5)
]
quadgrams_outlier_zscore["trigram"].to_csv("quadgrams_outlier_zscore.csv", index=False)


left_threshold = quadgrams_df["robust_z_score"].quantile(0.25) - 1.5 * (quadgrams_df["robust_z_score"].quantile(0.75) - quadgrams_df["robust_z_score"].quantile(0.25))
right_threshold = quadgrams_df["robust_z_score"].quantile(0.75) + 1.5 * (quadgrams_df["robust_z_score"].quantile(0.75) - quadgrams_df["robust_z_score"].quantile(0.25))
plt.figure(figsize=(24, 6))
sns.kdeplot(quadgrams_df["robust_z_score"], color="blue")
plt.axvline(x=0, color="r", linestyle="--")
plt.axvline(x = left_threshold, color="g", linestyle="--")
plt.axvline(x=right_threshold, color="g", linestyle="--")
plt.xlabel("Z-Score")
plt.ylabel("Density")
plt.show()

quadgrams_outlier_robust_zscore = trigrams_df[
    (quadgrams_df["robust_z_score"] > right_threshold) | (quadgrams_df["robust_z_score"] < left_threshold)
]
quadgrams_outlier_robust_zscore["quadgram"].to_csv("quadgrams_outlier_robust_zscore.csv", index=False)

plt.figure(figsize=(24, 6))
sns.kdeplot(quadgrams_df["mahalanobis_distance"], color="blue")
plt.axvline(x=0.8, color="r", linestyle="--")
plt.xlabel("Mahalanobis Distance")
plt.ylabel("Density")
plt.title("Quadgrams Mahalanobis Distance Distribution")
plt.show()
