import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from adjustText import adjust_text

df = pd.read_csv("all-ages.csv")
print(f"Dataset Loaded. Shape: {df.shape}")

df.columns = df.columns.str.strip().str.upper()

required_cols = ['MAJOR', 'MAJOR_CATEGORY', 'EMPLOYED', 'EMPLOYED_FULL_TIME_YEAR_ROUND',
                 'UNEMPLOYED', 'UNEMPLOYMENT_RATE', 'MEDIAN', 'TOTAL']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in dataset.")

df = df.dropna(subset=['EMPLOYED', 'EMPLOYED_FULL_TIME_YEAR_ROUND', 'UNEMPLOYMENT_RATE', 'MEDIAN', 'MAJOR_CATEGORY'])

df['EMPLOYMENT_RATE'] = df['EMPLOYED'] / df['TOTAL']

encoder = OneHotEncoder(sparse_output=False)
category_encoded = encoder.fit_transform(df[['MAJOR_CATEGORY']])
category_cols = encoder.get_feature_names_out(['MAJOR_CATEGORY'])
df_category = pd.DataFrame(category_encoded, columns=category_cols, index=df.index if 'MAJOR' in df.columns else df.index)

selected_features = ['MEDIAN', 'EMPLOYMENT_RATE']  # numeric features
scaler = MinMaxScaler()
X_scaled_numeric = scaler.fit_transform(df[selected_features])

X_combined = np.hstack([X_scaled_numeric, category_encoded])

df = df.set_index('MAJOR')
similarity_matrix = cosine_similarity(X_combined)
sim_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

query_majors = ['ECONOMICS', 'INFORMATION SCIENCES', 'HISTORY']
query_colors = {'ECONOMICS':'#E63946', 'INFORMATION SCIENCES':'#457B9D', 'HISTORY':'#2A9D8F'}

for q in query_majors:
    if q not in df.index:
        print(f"Warning: {q} not found in dataset!")

similarity_results = {}

for query in query_majors:
    top_similar = sim_df[query].sort_values(ascending=False)[1:11]
    similarity_results[query] = top_similar

    plt.figure(figsize=(8,5))
    palette = sns.light_palette(query_colors[query], n_colors=10, reverse=True)
    for i, (major, sim_val) in enumerate(top_similar.items()):
        plt.barh(major, sim_val, color=palette[i])
        plt.text(sim_val + 0.002, i, f"{sim_val:.3f}", va='center')

    plt.title(f"Top 10 Majors Similar to {query}")
    plt.xlabel("Cosine Similarity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    top_majors = df.loc[top_similar.index].copy()
    colors = sns.light_palette(query_colors[query], n_colors=10, reverse=True)

    plt.figure(figsize=(8,6))
    texts = []

    for i, major in enumerate(top_majors.index):
        x = top_majors.loc[major,'EMPLOYMENT_RATE']*100
        y = top_majors.loc[major,'MEDIAN']
        plt.scatter(x, y, color=colors[i], s=150)
        texts.append(plt.text(x + 0.2, y, major, fontsize=9))

    xq = df.loc[query,'EMPLOYMENT_RATE']*100
    yq = df.loc[query,'MEDIAN']
    plt.scatter(xq, yq, color=query_colors[query], s=250, marker='*', label=f"{query} (query)")
    texts.append(plt.text(xq + 0.2, yq, query, fontsize=10, fontweight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel("Employment Rate (%)")
    plt.ylabel("Median Salary ($)")
    plt.title(f"{query} & Top 10 Similar Majors: Employment vs Median Salary")
    plt.tight_layout()
    plt.show()

    print(f"\nTop 10 Similar Majors to {query} (Cosine Similarity, Median Salary $, Employment Rate %):")
    summary = pd.DataFrame({
        'Cosine Similarity': top_similar,
        'Median Salary ($)': df.loc[top_similar.index,'MEDIAN'],
        'Employment Rate (%)': df.loc[top_similar.index,'EMPLOYMENT_RATE']*100
    })
    summary = summary.round({'Cosine Similarity':3, 'Median Salary ($)':0, 'Employment Rate (%)':1})
    print(summary)

plt.figure(figsize=(10,7))
texts = []

for query in query_majors:
    top_similar = similarity_results[query]
    top_majors = df.loc[top_similar.index].copy()
    colors = sns.light_palette(query_colors[query], n_colors=10, reverse=True)

    for i, major in enumerate(top_majors.index):
        x = top_majors.loc[major,'EMPLOYMENT_RATE']*100
        y = top_majors.loc[major,'MEDIAN']
        plt.scatter(x, y, color=colors[i], s=100)

    xq = df.loc[query,'EMPLOYMENT_RATE']*100
    yq = df.loc[query,'MEDIAN']
    plt.scatter(xq, yq, color=query_colors[query], s=250, marker='*', label=f"{query} (query)")
    texts.append(plt.text(xq + 0.2, yq, query, fontsize=10, fontweight='bold'))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.xlabel("Employment Rate (%)")
plt.ylabel("Median Salary ($)")
plt.title("Top 10 Similar Majors for Each Query & Query Majors")
plt.legend()
plt.tight_layout()
plt.show()
