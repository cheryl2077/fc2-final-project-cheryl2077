import matplotlib.pyplot as plt
import numpy as np
def plot_mst_graph(mst_edges, clusters, layout="circle"):

    nodes = [n for g in clusters for n in g]
    N = len(nodes)

    # Node degrees for sizing
    degree = {n: 0 for n in nodes}
    for a, b, c, dist in mst_edges:
        degree[a] += 1
        degree[b] += 1

    # Layout selection

    # cite the below 2 lines from AI
    if layout == "circle":
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pos = {nodes[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

    else:
      # Spring layout with minimal repulsion
      pos = {n: np.random.rand(2) * 2 - 1 for n in nodes}

      for _ in range(200):

          # repulsion between all nodes
          # edited the below part with AI

          for i in range(N):
              ni = nodes[i]
              xi, yi = pos[ni]
              for j in range(i + 1, N):
                  nj = nodes[j]
                  xj, yj = pos[nj]

                  dx, dy = xi - xj, yi - yj
                  d2 = dx*dx + dy*dy + 1e-6
                  rep = 0.002 / d2    # small repulsive force

                  # push nodes apart
                  pos[ni] = (xi + rep * dx, yi + rep * dy)
                  pos[nj] = (xj - rep * dx, yj - rep * dy)

          # Original Attractive Forces
          # edited the below part with AI

          for a, b, c, dist in mst_edges:
              x1, y1 = pos[a]
              x2, y2 = pos[b]

              dx, dy = x2 - x1, y2 - y1
              d = np.sqrt(dx*dx + dy*dy) + 1e-6
              force = 0.02 * (1 - abs(c))

              pos[a] = (x1 + force * dx / d, y1 + force * dy / d)
              pos[b] = (x2 - force * dx / d, y2 - force * dy / d)

    # Colors per cluster (same as your original)
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
              'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    node_color = {}
    for idx, g in enumerate(clusters):
        for name in g:
            node_color[name] = colors[idx % len(colors)]

    plt.figure(figsize=(6, 6))

    # Draw edges
    # edited the below part with AI
    for a, b, c, dist in mst_edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]

        strength = abs(c)
        lw = 0.5 + 2 * strength
        alpha = max(0.25, strength)

        plt.plot([x1, x2], [y1, y2],
                 color='gray', linewidth=lw, alpha=alpha)

        # Draw correlation label at midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(mx, my, f"{c:.2f}",
                 fontsize=7, color="black", alpha=0.85)

    # Draw nodes
    for name, (x, y) in pos.items():
        node_size = 30 + degree[name] * 20
        plt.scatter(x, y, s=node_size, color=node_color[name])
        plt.text(x, y, name, fontsize=7, ha='center', va='center')

    plt.title("MST (Enhanced Layout + Correlation Encoding + Edge Labels)")
    plt.axis("off")
    plt.tight_layout()
    # plt.show()

def plot_cluster_time_series(price_df, clusters, normalize=False):

    # Plot time-series for each cluster with optional normalization (scale series to start at 1)

    figs = []  # store one figure per cluster

    for i, group in enumerate(clusters):
        fig = plt.figure(figsize=(8, 4))  # create a new figure for each cluster

        # collect data for computing average
        series_list = []

        for ticker in group:
            if ticker in price_df.columns:
                s = price_df[ticker]

                # Normalization option
                # edited the below line with AI
                if normalize:
                    s = (s - s.iloc[0]) / s.iloc[0]

                plt.plot(price_df.index, s, label=ticker, alpha=0.6)
                series_list.append(s.values)

        # Plot cluster-average curve
        if len(series_list) > 0:
            avg_series = np.mean(series_list, axis=0)
            plt.plot(
                price_df.index,
                avg_series,
                color='black',
                linewidth=2.2,
                label="Cluster Avg"
            )

        plt.title(f"Cluster {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Price" if not normalize else "Normalized Price")
        plt.legend()
        plt.tight_layout()

        figs.append(fig)

    return figs
