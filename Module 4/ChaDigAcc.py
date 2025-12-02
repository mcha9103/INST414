import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

df = pd.read_csv("Global_Development_Indicators_2000_2020.csv")

df = df[df["country_name"].apply(lambda x: "dividend" not in x.lower())]
df = df[df["country_name"].apply(lambda x: "income" not in x.lower())]

df_year = df[df["year"] == 2019].copy()

df_clean = df_year[["country_name", "internet_usage_pct", "mobile_subscriptions_per_100", "gdp_per_capita"]].dropna()
df_clean = df_clean.rename(columns={
    "country_name": "country",
    "internet_usage_pct": "internet_users",
    "mobile_subscriptions_per_100": "mobile_subs"})

features = ["internet_users", "mobile_subs", "gdp_per_capita"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

wcss = []
K = range(2, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker="o")
plt.title("Elbow Method for Optimal k (Year 2019)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()

sil_scores = {}
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

print("\nSilhouette scores:")
for k, s in sil_scores.items():
    print(f"k = {k}: {round(s, 3)}")

plt.figure(figsize=(8, 5))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
plt.title("Silhouette Score by Number of Clusters (2019)")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

k = 4  #optimal k

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_clean["cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 7))

cluster_colors = {
    0: "#e41a1c",  
    1: "#ff7f00",  
    2: "#e6b800",
    3: "#4daf4a"}

colors = df_clean["cluster"].map(cluster_colors)

scatter = plt.scatter(
    df_clean["internet_users"],      
    df_clean["gdp_per_capita"],
    c=colors,
    s=120,
    edgecolor="white",
    linewidth=0.7)

plt.xlabel("Internet Penetration (% of Population)", fontsize=12)
plt.ylabel("GDP per Capita (USD)", fontsize=12)
plt.title("Digital Inequality Clusters (2019): Internet Access vs GDP per Capita", fontsize=14)

plt.grid(True, alpha=0.25)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='white', label=f"Cluster {c}",
           markerfacecolor=cluster_colors[c], markersize=10)
    for c in sorted(df_clean["cluster"].unique())]

plt.legend(
    handles=legend_elements,
    title="Cluster",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=False)

plt.tight_layout()
plt.show()

cluster_summary = df_clean.groupby("cluster")[features].mean().round(2)
print("\n\n===== CLUSTER SUMMARY (MEAN VALUES, 2019) =====\n")
print(cluster_summary)

print("\n\n===== EXAMPLE COUNTRIES PER CLUSTER (2019) =====\n")

for c in range(k):
    countries = (
        df_clean[df_clean["cluster"] == c]
        .sort_values("internet_users", ascending=False)["country"]
        .head(5)
        .tolist())
    print(f"Cluster {c}: {countries}")
