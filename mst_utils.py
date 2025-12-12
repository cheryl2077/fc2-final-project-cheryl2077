import numpy as np
from data_utils import *
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

        return True


def build_mst_from_corr(corr_matrix):
    # Build an MST using distance = 1 - correlation
    # higher correlation, smaller distance
    names = list(corr_matrix.columns)
    edges = []

    # collect distances
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            # cite the below line from AI
            c = corr_matrix.loc[a, b]
            edges.append((1 - c, a, b, c))

    edges.sort(key=lambda x: x[0])

    uf = UnionFind(names)
    mst = []

    for dist, a, b, c in edges:
        if uf.union(a, b):
            mst.append((a, b, c, dist))

    return mst


def compute_mst_metrics(mst_edges):

    # Compute a few simple metrics that summarize the MST structure.

    if not mst_edges:
        # empty MST, return NaNs to avoid crashing downstream
        return {
            "tree_length": np.nan,
            "avg_correlation": np.nan,
            "max_degree": np.nan,
            "min_degree": np.nan,
        }

    # total distance of the tree
    # the lower the tree_length, the tighter the whole groups of stocks are
    tree_length = sum(dist for (a, b, c, dist) in mst_edges)

    # average correlation on MST edges
    avg_corr = float(np.mean([c for (a, b, c, dist) in mst_edges]))

    # degree of each node (number of edges connected to this node)
    # which stock is the center, which are the leafs
    degree = {}
    for a, b, c, dist in mst_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    max_deg = max(degree.values())
    min_deg = min(degree.values())

    metrics = {
        "tree_length": tree_length,
        "avg_correlation": avg_corr,
        "max_degree": max_deg,
        "min_degree": min_deg,
    }
    return metrics


def bootstrap_mst_stability(price_df, B=50, random_state=0):
    """
    Estimate how stable MST edges are under resampling.
    We bootstrap the time dimension, rebuild the MST each time,
    and record how often each edge shows up.
    """

    # use numpy's global RNG for simplicity
    np.random.seed(random_state)

    tickers = list(price_df.columns)
    edge_counts = {}

    for _ in range(B):
        # bootstrap the dates with replacement
        idx = np.random.choice(len(price_df), size=len(price_df), replace=True)
        sampled_prices = price_df.iloc[idx]

        # compute returns and correlation on each bootstrap sample
        sampled_returns = compute_log_returns(sampled_prices)
        corr_bs = sampled_returns.corr()

        # build MST on the bootstrapped correlation matrix
        mst_bs = build_mst_from_corr(corr_bs)

        # count how often each undirected edge appears
        # ie. if edge (AAPL-MSFT) appears in this sample, count += 1
        for a, b, c, dist in mst_bs:
            # sort tickers so ('A','B') and ('B','A') are treated the same
            key = tuple(sorted((a, b)))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    # convert counts to frequencies
    rows = []
    for (a, b), count in edge_counts.items():
        rows.append({
            "a": a,
            "b": b,
            "frequency": count / B
        })

    stability_df = pd.DataFrame(rows)
    return stability_df


# After using MST to keep the most important edges
# I want to cut the weak edges, only keep the strong edges and group them together

def clusters_from_mst(mst_edges,
                      corr_threshold=None,
                      dynamic=True,
                      quantile=0.6):          # quantile for dynamic threshold (keep 40% strongest edges, cut the rest 60%)

    # If corr_threshold is not provided, choose dynamically
    # generated this protection session with the help of AI
    if corr_threshold is None:
        if dynamic:
            # gather all correlations c from MST edges
            corr_values = [c for (_, _, c, _) in mst_edges]
            # quantile threshold
            corr_threshold = np.quantile(corr_values, quantile)
        else:
            corr_threshold = 0.5

    # Build adjacency list for graph
    graph = {}
    nodes = set()

    for a, b, c, dist in mst_edges:
        nodes.add(a)
        nodes.add(b)

        # keep only edges above threshold
        if c >= corr_threshold:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)

    # ensure every node exists in adjacency list
    for n in nodes:
        graph.setdefault(n, [])

    # DFS to find connected components
    # As long as two nodes are connected with a strong edge, they belong to one cluster

    visited = set()
    clusters = []

    for n in nodes:
        if n not in visited:
            stack = [n]
            comp = []
            visited.add(n)

            while stack:
                x = stack.pop()
                comp.append(x)
                for y in graph[x]:
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)

            clusters.append(comp)


    # Compute cluster quality metrics

    cluster_info = []

    # build a quick look-up table for raw correlations
    # MST only gives edges, but we need pairwise corr inside cluster
    # We reconstruct a correlation dict from MST edges.
    corr_lookup = {}
    for a, b, c, dist in mst_edges:
        corr_lookup[tuple(sorted((a, b)))] = c

    # compute metrics for each cluster
    for comp in clusters:
        if len(comp) == 1:
            # size-1 cluster metrics are trivial
            cluster_info.append({
                "members": comp,
                "size": 1,
                "avg_corr": np.nan,
                "min_corr": np.nan,
                "max_corr": np.nan
            })
            continue

        # debuged the below part with AI
        pair_corrs = []
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                a, b = comp[i], comp[j]
                key = tuple(sorted((a, b)))
                if key in corr_lookup:
                    pair_corrs.append(corr_lookup[key])
                # if not in MST edges, we simply ignore (MST is sparse)

        if len(pair_corrs) == 0:
            # no internal edges found (unlikely but safe)
            avg_corr = min_corr = max_corr = np.nan
        else:
            avg_corr = float(np.mean(pair_corrs))
            min_corr = float(np.min(pair_corrs))
            max_corr = float(np.max(pair_corrs))

        cluster_info.append({
            "members": comp,
            "size": len(comp),
            "avg_corr": avg_corr,
            "min_corr": min_corr,
            "max_corr": max_corr
        })

    # Return dictionary structure
    return {
        "clusters": clusters,
        "cluster_info": cluster_info,
        "threshold_used": corr_threshold
    }

