# 🔍 Search Query Anomaly Detection

This project focuses on analyzing and detecting anomalies in search query performance metrics using basic machine learning techniques and statistical analysis. By identifying unusual patterns in clicks, impressions, CTR (Click-Through Rate), and average position, the model helps uncover potentially abnormal or unexpected search behavior.

---

## 📊 Features

- **Data Cleaning**: Preprocesses the raw search query data, including parsing and normalizing the CTR values.
- **Text Analysis**: Extracts and visualizes the top 20 most frequently used words in search queries.
- **Exploratory Data Visualization**:
  - Top queries by **Clicks**, **Impressions**, and **CTR**.
  - Bottom queries by **CTR**.
  - **Correlation matrix** between performance metrics.
- **Anomaly Detection**:
  - Implements the **z-score method** to detect statistical outliers across key performance indicators.
  - Highlights anomalous queries that deviate significantly from the dataset norm.

---

## 📁 Dataset

The model uses a CSV file named `Queries.csv` that contains the following columns:

- `Top queries`
- `Clicks`
- `Impressions`
- `CTR` (Click-Through Rate as a percentage)
- `Position` (Average search result position)

> 📌 **Note**: Ensure the dataset is placed in the same directory as the script for proper execution.

---

## 🧠 Methodology

- **Z-Score Analysis**: For each numerical column, the z-score is computed. Queries with z-scores exceeding a defined threshold (default: `3`) in any of the metrics are flagged as anomalies.
- **Visualization Tools**: Plots and heatmaps are generated using `matplotlib` to gain insights and support anomaly interpretation.

---

## 📈 Output

- Graphs visualizing:
  - Word frequency
  - Top queries by Clicks, Impressions, CTR
  - Bottom queries by CTR
  - Correlation heatmap
- A filtered list of anomalous search queries with their metrics.

---

## 🛠️ Dependencies

- `pandas`
- `numpy`
- `matplotlib`

Install them using:

```bash
pip install pandas numpy matplotlib
```

## Usage

Run the script using any Python 3 environment:

```bash
python search_query_anomaly_detection.py
```

## 📌 Note

This project is intended for educational or exploratory data analysis purposes. For production-grade anomaly detection, consider integrating machine learning models like Isolation Forest, One-Class SVM, or deep learning-based approaches.
