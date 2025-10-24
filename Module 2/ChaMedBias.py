import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

df = pd.read_csv("allsides.csv")

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df[["name", "bias", "agree", "disagree", "agree_ratio"]].dropna(subset=["name", "bias"])

excluded = ["AllSides", "Ad Fontes", "Media Bias/Fact Check"]
df = df[~df["name"].str.contains("|".join(excluded), case=False, na=False)]

df["bias"] = df["bias"].str.lower().str.strip()
df["agree_ratio"] = (df["agree_ratio"] - df["agree_ratio"].min()) / (df["agree_ratio"].max() - df["agree_ratio"].min())
df["total_votes"] = df["agree"] + df["disagree"]

df = df.sort_values(by="total_votes", ascending=False).head(40).reset_index(drop=True)

G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["name"], bias=row["bias"], agree_ratio=row["agree_ratio"])

#Bias order and numeric mapping for weighting
bias_order = ["left", "left-center", "center", "right-center", "right"]
bias_numeric = {b: i for i, b in enumerate(bias_order)}

#Weighted inter-bias edges
outlets = df[["name", "bias", "agree_ratio"]].values
for i in range(len(outlets)):
    for j in range(i + 1, len(outlets)):
        name_i, bias_i, agree_i = outlets[i]
        name_j, bias_j, agree_j = outlets[j]
        if bias_i != bias_j:
            #Smaller agree_ratio difference = stronger connection
            agree_sim = 1 - abs(agree_i - agree_j)
            #Bias distance (closer ideologically = stronger connection)
            bias_dist = abs(bias_numeric.get(bias_i, 0) - bias_numeric.get(bias_j, 0))
            bias_factor = 1 / (1 + bias_dist)

            weight = agree_sim * bias_factor
            if weight > 0.3:  #Threshold for meaningful similarity
                G.add_edge(name_i, name_j, weight=weight, type="inter")

print(f"\nNetwork built with {len(G.nodes())} nodes and {len(G.edges())} weighted edges.")

betweenness = nx.betweenness_centrality(G, weight="weight")
nx.set_node_attributes(G, betweenness, "betweenness")

bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n--- Top 10 Bridge Outlets ---")
for outlet, score in bridges:
    print(f"{outlet}: {score:.4f}")

color_map = {
    "left": "#08306b",
    "left-center": "#4292c6",
    "center": "#bdbdbd",
    "right-center": "#fb6a4a",
    "right": "#99000d"
}

node_colors = [color_map.get(G.nodes[n]["bias"], "gray") for n in G.nodes()]

edge_colors = ["orange" for _ in G.edges()]
edge_widths = [G.edges[e]["weight"] * 2 for e in G.edges()]  

bias_xpos = {b: i for i, b in enumerate(bias_order)}
pos = {}
for n, data in G.nodes(data=True):
    bias = data["bias"]
    pos[n] = np.array([
        bias_xpos.get(bias, 0) + np.random.normal(0, 0.2),
        np.random.normal(0, 1)
    ])

max_bet = max(betweenness.values()) if betweenness else 1
node_sizes = [200 + 3000 * (betweenness[n] / max_bet) for n in G.nodes()]

plt.figure(figsize=(14, 9))
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color=edge_colors, width=edge_widths)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=7, font_color="black")

plt.title("Weighted Network of U.S. Media Outlets by Bias (Top 40 Outlets)", fontsize=15)
plt.axis("off")

legend_handles = [Patch(color=color_map[b], label=b.capitalize()) for b in bias_order if b in df["bias"].unique()]
plt.legend(handles=legend_handles, title="Bias", loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

bridge_df = pd.DataFrame(bridges, columns=["Outlet", "Betweenness"])
plt.figure(figsize=(10, 5))
plt.barh(bridge_df["Outlet"], bridge_df["Betweenness"], color="#6baed6")
plt.gca().invert_yaxis()
plt.title("Top 10 Bridge Media Outlets (Weighted Betweenness Centrality)", fontsize=13)
plt.xlabel("Weighted Betweenness Centrality (Bridging Score)")
plt.tight_layout()
plt.show()

summary = pd.DataFrame({
    "Metric": ["Nodes", "Edges", "Avg Degree", "Density"],
    "Value": [
        len(G.nodes()),
        len(G.edges()),
        np.mean([d for _, d in G.degree()]),
        nx.density(G)
    ]
})
print("\n--- Summary ---")
print(summary)

nx.set_node_attributes(G, {n: bias_numeric[G.nodes[n]["bias"]] for n in G.nodes()}, "bias_numeric")
assort = nx.attribute_assortativity_coefficient(G, "bias_numeric")
print(f"\nBias Assortativity (Polarization Measure): {assort:.3f}")

if assort > 0.25:
    interpretation = "High polarization (like-minded groups strongly connected)"
elif assort > 0.0:
    interpretation = "Moderate polarization"
elif assort > -0.2:
    interpretation = "Mild cross-bias connectivity"
else:
    interpretation = "Low polarization or strong bridging"

print(f"Interpretation: {interpretation}")
