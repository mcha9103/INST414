import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Financial Statements.csv")

df.rename(columns=lambda x: x.strip(), inplace=True)
df = df.dropna(subset=["Market Cap(in B USD)"]) 
df = df.sort_values(["Company", "Year"])
df["MarketCap_Growth"] = df.groupby("Company")["Market Cap(in B USD)"].pct_change().shift(-1)

target = "MarketCap_Growth"

numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
correlations = corr_matrix[target].sort_values(ascending=False)

print("\nTop correlated financial ratios with Market Cap:")
print(correlations.head(10))
print("\nLeast correlated financial ratios with Market Cap:")
print(correlations.tail(10))

profitability_vars = ["ROE", "ROA", "ROI", "Net Profit Margin"]
liquidity_vars = ["Current Ratio"]
leverage_vars = ["Debt/Equity Ratio"]
efficiency_vars = ["Earning Per Share", "Free Cash Flow per Share"]

group_cols = ["Company", "Year"]
df_grouped = df[group_cols + profitability_vars + liquidity_vars + leverage_vars + efficiency_vars + ["MarketCap_Growth"]].copy()

df_grouped = df_grouped.dropna(subset=["MarketCap_Growth"])

plt.figure(figsize=(10,6))
corr = df_grouped.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Between Financial Ratios and Next-Year Market Cap Growth")
plt.tight_layout()
plt.show()

categories = {
    "Profitability": profitability_vars,
    "Liquidity": liquidity_vars,
    "Leverage": leverage_vars,
    "Efficiency": efficiency_vars
}
results = {}

for category, vars_ in categories.items():
    data = df_grouped.dropna(subset=vars_ + ["MarketCap_Growth"])
    X = data[vars_]
    y = data["MarketCap_Growth"]

    if len(X) > 1:
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        results[category] = r2

print("\nR² Scores by Category (Predicting Next-Year Market Cap Growth):")
for cat, score in results.items():
    print(f"{cat}: {score:.3f}")
    
plt.figure(figsize=(6,4))
sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
sns.barplot(x=list(sorted_results.keys()), y=list(sorted_results.values()))
plt.title("Predictive Strength of Financial Categories (R²)")
plt.ylabel("R² (Higher = Better Prediction)")
plt.tight_layout()
plt.show()
