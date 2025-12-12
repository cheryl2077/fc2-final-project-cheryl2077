import streamlit as st
import matplotlib.pyplot as plt
from data_utils import (load_price_data,
    compute_log_returns,
    compute_correlation_matrix
                        )
from mst_utils import (
    build_mst_from_corr,
    compute_mst_metrics,
    bootstrap_mst_stability,
    clusters_from_mst,
)
from plot_utils import plot_mst_graph, plot_cluster_time_series

st.title("Stock MST Explorer")

user_input = st.text_input(
    "Enter tickers separated by commas (e.g. AAPL, MSFT, NVDA, AMD, META, JNJ, "
    "PG, KO, PEP, WMT, JPM, BAC, GS, MS, XOM, CVX, COP, NEE, DUK, UNH)"
)

# Page config
# ============================================================
st.set_page_config(
    page_title="Stock MST Explorer",
    layout="wide"
)

st.title("📈 Stock MST Explorer")
st.caption(
    "Explore correlation-based minimum spanning trees and cluster structure "
    "in equity markets"
)

# Sidebar: user inputs
# ============================================================
with st.sidebar:
    st.header("Inputs")

    user_input = st.text_area(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA, AMD, META, JNJ, PG, KO, PEP, WMT, JPM, BAC, GS, MS, XOM, CVX, COP, NEE, DUK, UNH",
        height=100
    )

    normalize = st.checkbox(
        "Normalize price series",
        value=True,
        help="Scale each price series to start from a common baseline"
    )

    st.markdown("---")
    st.caption("MST & clustering settings")


if user_input:
    tickers = [t.strip().upper() for t in user_input.split(",")]
    st.write("You selected:", tickers)

    # 1️. Load data
    price_df = load_price_data(tickers)
    st.dataframe(price_df.head())

    # 2️. Returns & correlation
    returns = compute_log_returns(price_df)
    corr = compute_correlation_matrix(returns)
    st.write("Correlation matrix")
    st.dataframe(corr)

    # 3️. MST
    mst = build_mst_from_corr(corr)
    st.write("MST edges")
    for a, b, c, dist in mst:
        st.write(f"{a} — {b} | corr={c:.2f}, dist={dist:.2f}")

    # 4️. MST metrics
    metrics = compute_mst_metrics(mst)
    st.write("MST metrics")
    st.json(metrics)

    # 5️. Bootstrap stability
    st.write("Bootstrap MST stability")
    stability_df = bootstrap_mst_stability(price_df, B=50, random_state=42)
    st.dataframe(stability_df)

    # 6️. Clustering
    clusters_dict = clusters_from_mst(mst, dynamic=True, quantile=0.7)
    clusters = clusters_dict["clusters"]
    cluster_info = clusters_dict["cluster_info"]

    st.write("Clusters:", clusters)
    st.write("Cluster info:")
    for info in cluster_info:
        st.write(info)

    # 7️. Visualization
    # MST circle
    plot_mst_graph(mst, clusters, layout="circle")
    st.pyplot(plt.gcf())
    plt.close()

    # MST spring
    plot_mst_graph(mst, clusters, layout="spring")
    st.pyplot(plt.gcf())
    plt.close()

    # Cluster time series
    figs = plot_cluster_time_series(
        price_df,
        clusters,
        normalize=True
    )

    for fig in figs:
        st.pyplot(fig)
        plt.close(fig)





