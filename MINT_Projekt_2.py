import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

st.set_page_config(
    page_title="Dice Roll Dashboard",
    page_icon="🎲",
    layout="wide"
)



def run_experiment_e1():
    """
    Simulates Experiment (E1):
    Rolls a 6-sided die 20 times and returns the sum.
    """
    total_sum = 0
    for _ in range(20):
        total_sum += random.randint(1, 6)
    return total_sum


def gaussian_curve(x, mu, variance):
    """
    Calculates the value of the Gaussian curve (normal distribution)
    using the formula from the project.
    """
    sigma = math.sqrt(variance)
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -((x - mu) ** 2) / (2 * variance)
    return coefficient * math.exp(exponent)


st.sidebar.title("MINT Projekt 2")
st.sidebar.header("Simulation Controls")

num_experiments = st.sidebar.radio(
    "Choose your group (Number of Experiments):",
    (21, 42),
    captions=["Group G1 (21 runs)", "Group G2 (42 runs)"],
    index=1,  # Default to 42 (G2)
    horizontal=True
)

st.sidebar.button("Re-run Simulation", type="primary")

results = [run_experiment_e1() for _ in range(num_experiments)]

mu = 70  # Theoretical mean (20 * 3.5)
variance = 20 * (35 / 12)  # Theoretical variance
sigma = math.sqrt(variance)

actual_mean = np.mean(results)
actual_variance = np.var(results)

st.title("🎲 Würfeln und der Zentrale Grenzwertsatz")
st.markdown(f"Displaying results for **{num_experiments}** experiments (Group G{1 if num_experiments == 21 else 2}).")

dashboard_tab, info_tab = st.tabs(["📊 Dashboard", "ℹ️ Project Info"])

with dashboard_tab:
    st.header("Statistical Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Theoretical Stats")
        st.metric("Mean (μ)", f"{mu:.2f}")
        st.metric("Variance (σ²)", f"{variance:.2f}")
    with col2:
        st.subheader("Simulation Stats")
        st.metric("Actual Mean", f"{actual_mean:.2f}")
        st.metric("Actual Variance", f"{actual_variance:.2f}")

    st.header("Histogram vs. Gaussian Curve")

    fig, ax = plt.subplots(figsize=(10, 6))

    class_width = 5
    bins = np.arange(30, 111, class_width)

    ax.hist(results, bins=bins, density=True, alpha=0.7,
            edgecolor='black', label=f'Histogram of {num_experiments} Results')

    x_curve = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y_curve = [gaussian_curve(val, mu, variance) for val in x_curve]
    ax.plot(x_curve, y_curve, 'r-', linewidth=2,
            label=f'Gaussian Curve (μ=70, σ²={variance:.2f})')

    ax.set_title("Dice Sum Distribution")
    ax.set_xlabel("Sum of 20 Dice Rolls")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)

    st.markdown("""
    **Observation:**
    Notice how the histogram (blue bars) fits the theoretical Gaussian curve (red line).
    * With 21 experiments (G1), the fit is usually rough.
    * With 42 experiments (G2), the fit is noticeably better.
    """)

    st.header(f"Raw Data for {num_experiments} Experiments")
    st.markdown(
        "Here are the individual sums from each experiment. You can sort the table by clicking the column headers.")

    df = pd.DataFrame({
        "Experiment #": range(1, num_experiments + 1),
        "Sum of 20 Dice": results
    })

    st.dataframe(df, height=500, use_container_width=True)

# --- This tab contains the project info ---
with info_tab:
    st.header("About The Project")
    st.markdown("""
    ##    """)
