# ⚽ FIFA Player Performance & Market Value Analysis

An end-to-end data science project analyzing FIFA player performance data — covering exploratory data analysis, preprocessing, visualization, KNN classification, and feature engineering. Built in **Python** on **Google Colab**.

## About

This project uses the [FIFA Player Performance & Market Value Analytics](https://www.kaggle.com/datasets/jayjoshi37/fifa-player-performance-and-market-value-analytics) dataset from Kaggle (2,800 players) to explore what drives player market value and predict transfer risk levels using machine learning.

The notebook is structured in 5 parts following a complete data science workflow — from raw data exploration to predictive modeling.

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — FIFA Player Performance & Market Value Analytics |
| Records | 2,800 players |
| Features | 15+ (age, overall rating, potential, goals, assists, market value, position, etc.) |
| Target Variable | `transfer_risk_level` (Low / Medium / High) |

## Project Structure

### Part 1 — Data Exploration
- First/last rows inspection, shape, dtypes, and dataset info
- Target variable (`transfer_risk_level`) class distribution analysis
- Categorical feature analysis (`position` — unique values, frequency)
- Numerical feature statistics (mean, median, std, percentiles)
- Missing value detection and duplicate record removal

### Part 2 — Data Preprocessing
- Label encoding for categorical features
- Standard scaling for numerical features
- Chi-square test for feature significance (position vs transfer risk)
- Feature selection — removed non-predictive columns (`player_id`, `player_name`, `injury_prone`)
- Stratified sampling to maintain class proportions

### Part 3 — Data Visualization
- Histogram of overall rating distribution
- Boxplot of market value to identify outliers
- Correlation heatmap across all numerical features
- Bar chart of transfer risk level distribution
- Scatter plot of goals vs market value by transfer risk

### Part 4 — KNN Classification
- K-Nearest Neighbors classifier with cross-validation
- Train/test split with stratified sampling
- Model evaluation with accuracy, precision, recall, and F1-score
- Hyperparameter tuning for optimal K value
- Confusion matrix visualization

### Part 5 — Feature Engineering
- Created `performance_score` feature — (goals + assists) / matches played
- Captures player efficiency rather than raw totals
- Validated feature significance across transfer risk levels

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| matplotlib | Base plotting |
| seaborn | Statistical visualizations |
| scikit-learn | KNN, StandardScaler, LabelEncoder, train_test_split, cross-validation |
| Google Colab | Development environment |

## How to Run

1. **Open in Google Colab** — upload the `.ipynb` notebook file
2. **Upload the dataset** — download the CSV from [Kaggle](https://www.kaggle.com/datasets/jayjoshi37/fifa-player-performance-and-market-value-analytics) and upload to Colab
3. **Run all cells** sequentially (each part builds on the previous)

Or clone locally:
```bash
git clone https://github.com/IsmailSamir1/FIFA-Player-Performance-Analysis.git
cd FIFA-Player-Performance-Analysis
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

## Key Findings

- Market value is heavily right-skewed — most players cluster at lower valuations with few high-value outliers
- `overall_rating` and `potential_rating` are the strongest predictors of transfer risk
- The `performance_score` engineered feature (goals + assists per match) showed meaningful variation across transfer risk levels
- KNN classification achieved reliable predictions after hyperparameter tuning

## What I Learned

- Structuring a full data science pipeline from EDA to modeling
- Feature engineering to create more predictive variables from raw data
- Applying KNN classification with cross-validation and hyperparameter tuning
- Using stratified sampling to preserve class balance
- Statistical testing (Chi-square) for feature significance
- Creating clear, interpretable visualizations with matplotlib and seaborn

## Context

Built during **Semester 4** (Spring 2026) at the **German International University**, Cairo — as part of the Data Science course (Dr. Caroline Sabty).

## License

This project was developed for educational purposes at GIU.
