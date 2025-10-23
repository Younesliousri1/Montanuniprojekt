import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy.stats as stats

# --- Page Configuration ---
st.set_page_config(
    page_title="Central Limit Theorem Dashboard",
    page_icon="🎲",
    layout="wide"
)

# --- Core Experiment & Stat Functions ---

def run_experiment(num_rolls, die_sides):
    """
    Simulates 1 experiment:
    Rolls a dice `num_rolls` times and returns the sum.
    """
    return sum(random.randint(1, die_sides) for _ in range(num_rolls))

def calculate_theoretical_stats(num_rolls, die_sides):
    """Calcul de la moyenne et de la variance théoriques."""
    # Mean and variance of a single die roll
    mu_one = (die_sides + 1) / 2
    var_one = (die_sides**2 - 1) / 12
    
    # Total mean and variance for the sum of `num_rolls`
    total_mu = num_rolls * mu_one
    total_var = num_rolls * var_one
    return total_mu, total_var

def gaussian_curve(x, mu, variance):
    """Calculates the value of the Gaussian curve."""
    sigma = math.sqrt(variance)
    return (1 / (sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-((x - mu)**2) / (2 * variance))

# --- Sidebar (Controls) ---
st.sidebar.title("MINT Math Project 2")
st.sidebar.header("Simulation Controls")

# INTERACTIVE WIDGET for the "Additional Question"
dice_sides = st.sidebar.selectbox(
    "1. Dice Type (Sides)",
    (6, 12),
    index=0, # Default to 6
    help="Select the number of sides on the die. The project asks for 6, with an additional question about 12."
)

# The project specifies 20 rolls, but a slider is more insightful
num_rolls = st.sidebar.slider(
    "2. Rolls per Experiment (N)",
    1, 50, 20,
    help="Number of times to roll the die and sum the result in a *single* experiment. The project specifies 20."
)

# --- MODIFICATION: Replaced radio button with number_input ---
num_experiments = st.sidebar.number_input(
    "3. Number of Experiments (Samples)",
    min_value=1,
    value=42,  # Default to 42
    step=1,
    help="Enter the total number of experiments to run. The project specifies 21 (G1) or 42 (G2)."
)

st.sidebar.button("Re-run Simulation", type="primary")

# --- Run Simulation & Calculations ---

# 1. Run the simulation
results = [run_experiment(num_rolls, dice_sides) for _ in range(num_experiments)]

# 2. Get Theoretical stats
mu, variance = calculate_theoretical_stats(num_rolls, die_sides)
sigma = math.sqrt(variance)

# 3. Get Actual (Sample) stats
actual_mean = np.mean(results)
actual_var = np.var(results)
actual_median = np.median(results)
actual_skew = stats.skew(results)
actual_kurt = stats.kurtosis(results) # Fisher's kurtosis (normal=0)

# 4. Run Shapiro-Wilk Normality Test
# Test requires at least 3 samples
if num_experiments >= 3:
    shapiro_stat, p_value = stats.shapiro(results)
else:
    shapiro_stat, p_value = (None, None) # Set to None if test can't run

# --- Main Page Layout ---
st.title("🎲 Central Limit Theorem Dashboard")
st.markdown(f"Simulating **{num_experiments}** experiments, each summing **{num_rolls}** rolls of a **{die_sides}-sided** die.")

# --- Tabs for Content ---
tab_main, tab_stats, tab_info = st.tabs(
    ["📊 Main Plot", "📈 In-Depth Analysis", "ℹ️ Project Info"]
)

with tab_main:
    st.header("Histogram vs. Theoretical Gaussian Curve")

    # --- Create the Main Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot the scaled histogram
    # Make bins dynamic based on mean and sigma
    class_width = max(1, int(sigma / 3)) # Heuristic for class width
    min_val = int(mu - 4*sigma)
    max_val = int(mu + 4*sigma)
    bins = np.arange(min_val, max_val + class_width, class_width)
    
    ax.hist(results, bins=bins, density=True, alpha=0.7,
            edgecolor='black', label=f'Histogram of {num_experiments} Results')
    
    # 2. Plot the theoretical Gaussian curve
    x_curve = np.linspace(min_val, max_val, 300)
    y_curve = [gaussian_curve(val, mu, variance) for val in x_curve]
    ax.plot(x_curve, y_curve, 'r-', linewidth=2,
            label=f'Gaussian Curve (μ={mu:.2f}, σ²={variance:.2f})')
    
    ax.set_title("Distribution of Sums vs. Normal Distribution")
    ax.set_xlabel(f"Sum of {num_rolls} Dice Rolls")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

    st.header("The 'Before' Picture: Distribution of a Single Die Roll")
    st.markdown("The Central Limit Theorem is powerful because it takes a *uniform* distribution (like one die roll) and produces a *normal* distribution (a bell curve) when you sum many of them.")
    
    # Plot for a single die roll
    single_die_data = pd.DataFrame(
        {'Probability': [1/die_sides] * die_sides},
        index=range(1, die_sides + 1)
    )
    st.bar_chart(single_die_data)

with tab_stats:
    st.header("In-Depth Statistical Analysis")
    
    # --- 1. Key Metrics ---
    st.subheader("Theoretical vs. Actual Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Theoretical Mean (μ)", f"{mu:.3f}")
        st.metric("Theoretical Variance (σ²)", f"{variance:.3f}")
    with col2:
        st.metric("Actual Mean (from data)", f"{actual_mean:.3f}")
        st.metric("Actual Variance (from data)", f"{actual_var:.3f}")

    # --- 2. Advanced Shape Statistics ---
    st.subheader("Distribution Shape Statistics")
    cols = st.columns(3)
    cols[0].metric("Median", f"{actual_median:.3f}", 
                   help="The 50th percentile (middle value). For a perfect normal distribution, Mean == Median.")
    cols[1].metric("Skewness", f"{actual_skew:.3f}", 
                   help="Measures asymmetry. A normal distribution has a skew of 0. Negative = left tail, Positive = right tail.")
    cols[2].metric("Kurtosis (Fisher)", f"{actual_kurt:.3f}",
                   help="Measures 'tailedness' or peakedness. A normal distribution has a kurtosis of 0. Positive = sharper peak, Negative = flatter peak.")

    # --- 3. Normality Testing ---
    st.subheader("Professional Normality Tests")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Shapiro-Wilk Test")
        
        # --- MODIFICATION: Check if test could run ---
        if p_value is None:
            st.error("Test requires at least 3 experiments to run.")
        else:
            st.metric("p-value", f"{p_value:.4f}")
            if p_value > 0.05:
                st.success("**Conclusion:** The data appears to be normally distributed. (p > 0.05)")
                st.markdown("We *cannot reject* the null hypothesis that the data comes from a normal distribution.")
            else:
                st.warning("**Conclusion:** The data does *not* appear to be normally distributed. (p <= 0.05)")
                st.markdown("We *reject* the null hypothesis. This is common with small sample sizes.")
            st.markdown("*(Note: This test has low power with small samples, so it may fail even if the data looks normal.)*")
            
    with col2:
        st.subheader("Q-Q (Quantile-Quantile) Plot")
        # Create Q-Q plot
        fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
        stats.probplot(results, dist="norm", plot=ax_qq)
        ax_qq.set_title("Normal Q-Q Plot")
        ax_qq.set_xlabel("Theoretical Quantiles (Normal)")
        ax_qq.set_ylabel("Sample Quantiles (Your Data)")
        
        st.pyplot(fig_qq)
        st.markdown("If the data is normal, the blue dots should lie on the red line. Deviations from the line show non-normality.")
        
    # --- 4. Data Table ---
    with st.expander(f"Show Raw Data Table ({num_experiments} rows)"):
        df = pd.DataFrame({
            "Experiment #": range(1, num_experiments + 1),
            "Sum": results
        })
        st.dataframe(df, height=300, use_container_width=True)

with tab_info:
    st.header("About The Project")
    st.markdown("""
    This dashboard simulates the **MINT Projekt 2** on the **Central Limit Theorem (CLT)**.
    
    ### The Original Experiment (E1)
    * [cite_start]**Task:** Roll a 6-sided die 20 times and sum the results. [cite: 7]
    * [cite_start]**Group 1 ($G_1$):** Repeat this experiment 21 times. [cite: 6]
    * [cite_start]**Group 2 ($G_2$):** Repeat this experiment 42 times. [cite: 11]
    * [cite_start]**Goal:** Compare the resulting histogram to the specific Gaussian curve: [cite: 12]
        [cite_start]$$f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^{2}}} \\exp\\left(-\\frac{(x-70)^{2}}{2\\sigma^{2}}\\right)$$ [cite: 14]
        [cite_start]...where $\\mu = 70$ and $\\sigma^{2} = 20 \cdot \\frac{35}{12}$. [cite: 14, 16]
        
    ### This Dashboard
    [cite_start]This app allows you to run the simulation and also interactively explore the **Zusatzfrage (Additional Question)** by changing the die type from 6-sided to 12-sided. [cite: 21] You can also see the effect of changing the number of rolls (N) and the number of experiments (Samples).
    """)
