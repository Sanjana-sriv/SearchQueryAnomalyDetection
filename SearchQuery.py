import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
queries_df = pd.read_csv("Queries.csv")
print(queries_df.head())
print(queries_df.info())

# Cleaning CTR column
queries_df['CTR'] = queries_df['CTR'].str.rstrip('%').astype('float') / 100

# Function to clean and split the queries into words
def clean_and_split(query):
    words = "".join([char.lower() if char.isalpha() or char.isspace() else " " for char in query]).split()
    return words

# Split each query into words and count the frequency of each word
word_counts = {}
for query in queries_df['Top queries']:
    words = clean_and_split(query)
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

# Convert word counts to a DataFrame
word_freq_df = pd.DataFrame(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20], columns=['Word', 'Frequency'])

# Plotting the word frequencies
plt.figure(figsize=(12, 6))
plt.bar(word_freq_df['Word'], word_freq_df['Frequency'], color='blue')
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title("Top 20 Most Common Words in Search Queries")
plt.xticks(rotation=45)
plt.show()

# Top queries by Clicks and Impressions
top_queries_clicks_vis = queries_df.nlargest(10, 'Clicks')[['Top queries', 'Clicks']]
top_queries_impressions_vis = queries_df.nlargest(10, 'Impressions')[['Top queries', 'Impressions']]

# Plotting Top Queries by Clicks
plt.figure(figsize=(12, 6))
plt.bar(top_queries_clicks_vis['Top queries'], top_queries_clicks_vis['Clicks'], color='green')
plt.xlabel("Top Queries")
plt.ylabel("Clicks")
plt.title("Top Queries by Clicks")
plt.xticks(rotation=45)
plt.show()

# Plotting Top Queries by Impressions
plt.figure(figsize=(12, 6))
plt.bar(top_queries_impressions_vis['Top queries'], top_queries_impressions_vis['Impressions'], color='orange')
plt.xlabel("Top Queries")
plt.ylabel("Impressions")
plt.title("Top Queries by Impressions")
plt.xticks(rotation=45)
plt.show()

# Queries with highest and lowest CTR
top_ctr_vis = queries_df.nlargest(10, 'CTR')[['Top queries', 'CTR']]
bottom_ctr_vis = queries_df.nsmallest(10, 'CTR')[['Top queries', 'CTR']]

# Plotting Top Queries by CTR
plt.figure(figsize=(12, 6))
plt.bar(top_ctr_vis['Top queries'], top_ctr_vis['CTR'], color='purple')
plt.xlabel("Top Queries")
plt.ylabel("CTR")
plt.title("Top Queries by CTR")
plt.xticks(rotation=45)
plt.show()

# Plotting Bottom Queries by CTR
plt.figure(figsize=(12, 6))
plt.bar(bottom_ctr_vis['Top queries'], bottom_ctr_vis['CTR'], color='red')
plt.xlabel("Top Queries")
plt.ylabel("CTR")
plt.title("Bottom Queries by CTR")
plt.xticks(rotation=45)
plt.show()

# Correlation matrix visualization
correlation_matrix = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']].corr()
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix")
plt.show()

# Detecting anomalies using z-score method
def detect_anomalies(df, threshold=3):
    means = df.mean()
    stds = df.std()
    z_scores = (df - means) / stds
    anomalies = (np.abs(z_scores) > threshold).any(axis=1)
    return anomalies

queries_df['anomaly'] = detect_anomalies(queries_df[['Clicks', 'Impressions', 'CTR', 'Position']])

# Filtering out the anomalies
anomalies = queries_df[queries_df['anomaly']]
print(anomalies[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])
