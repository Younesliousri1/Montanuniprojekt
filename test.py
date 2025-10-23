import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy.stats as stats
import qrcode
from io import BytesIO

# --- Page Configuration (Must be the first st command) ---
st.set_page_config(
    page_title="Central Limit Theorem Dashboard",
    page_icon="🎲",
    layout="wide"
)

# --- 1. SET YOUR PUBLIC APP URL HERE ---
# Find this in your Streamlit Cloud "Manage app" menu.
# It should look like: https://your-app-name.streamlit.app
APP_URL = "https://your-app-name.streamlit.app" 


# --- Helper Function to Generate QR Code ---
def generate_qr_code(url):
    """Generates a QR code image from a URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save image to an in-memory buffer
    buf = BytesIO()
    img.save(buf)
    return buf.getvalue()

# --- Session State to Manage Pages ---
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

def start_simulation():
    """Callback function to switch to the main dashboard."""
    st.session_state.show_dashboard = True

# --- 2. WELCOME PAGE LOGIC ---
if not st.session_state.show_dashboard:
    
    st.title("🎲 Welcome to the MINT Projekt 2")
    st.title("Central Limit Theorem Dashboard")
    
    st.markdown("""
    This interactive dashboard is a simulation for the **MINT Projekt 2**, demonstrating the
    **Central Limit Theorem (CLT)** in action.
    
    - **Experiment:** See what happens when you roll a die 20 times and sum the results.
    - **Observe:** Watch how the histogram of these sums (your data) starts to form a perfect
      bell curve (the Gaussian distribution) as you run more experiments.
    - **Analyze:** Use the in-depth statistics to see how well your data matches the theory.
    """)
    
    st.button("Start Simulation", type="primary", on_click=start_simulation, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Share this App")
        st.markdown("Scan the QR code with your phone or tablet to open this app on your device.")
        if APP_URL == "https://your-app-name.streamlit.app":
            st.error("Please update the `APP_URL` variable in the Python script (Line 16) with your public Streamlit URL to generate the QR code.")
        else:
            try:
                qr_code_image = generate_qr_code(APP_URL)
                st.image(qr_code_image, caption="Scan to open this app", width=250)
            except Exception as e:
                st.error(f"Error generating QR code: {e}")

    with col2:
        st.subheader("About the Project")
        st.markdown("""
        - **Project:** MINT Projekt 2
        - **Topic:** Würfeln und der Zentrale Grenzwertsatz
        - **Controls:** Use the sidebar (on the next page) to change parameters like:
            - Die Type (6-sided or 12-sided)
            - Rolls per Experiment (N)
            - Number of Experiments (Samples)
        """)

# --- 3. MAIN DASHBOARD LOGIC (Your existing code) ---
else:
    # --- Core Experiment & Stat Functions ---
    def run_experiment(num_rolls, die_sides):
        """
        Simulates one experiment:
        Rolls a die `num_rolls` times and returns the sum.
        """
        return sum(random.randint(1, die_sides) for _ in range(num_rolls))

    def calculate_theoretical_stats(num_rolls, die_sides):
        """Calculates the theoretical mean and variance."""
        mu_one = (die_sides + 1) / 2
        var_one = (die_sides**2 - 1) / 12
        total_mu = num_rolls * mu_one
        total_var = num_rolls * var_one
        return total_mu, total_var

    def gaussian_curve(x, mu, variance):
        """Calculates the value of the Gaussian curve."""
        sigma = math.sqrt(variance)
        return (1 / (sigma * math.sqrt(2 * math.pi))) * \
               math.exp(-((x - mu)**2) / (2 * variance))

    # --- Sidebar (Controls) ---
    st.sidebar.title("MINT Projekt 2")
    st.sidebar.header("Simulation Controls")
    
    # Button to go back to the Welcome Page
    if st.sidebar.button("Back to Welcome Page"):
        st.session_state.show_dashboard = False
        st.rerun()

    die_sides = st.sidebar.selectbox(
        "1. Die Type (Sides)",
        (6, 12),
        index=0,
        help="Select the number of sides on the die. The project asks for 6, with an additional question about 12."
    )
    num_rolls = st.sidebar.slider(
        "2. Rolls per Experiment (N)",
        1, 50, 20,
        help="Number of times to roll the die and sum the result in a *single* experiment. The project specifies 20."
    )
    num_experiments = st.sidebar.number_input(
        "3. Number of Experiments (Samples)",
        min_value=1,
        value=42,
        step=1,
        help="Enter the total number of experiments to run. The project specifies 21 (G1) or 42 (G2)."
    )
    st.sidebar.button("Re-run Simulation", type="primary")

    # --- Run Simulation & Calculations ---
    results = [run_experiment(num_rolls, die_sides) for _ in range(num_experiments)]
    mu, variance = calculate_theoretical_stats(num_rolls, die_sides)
    sigma = math.sqrt(variance)
    actual_mean = np.mean(results)
    actual_var = np.var(results)
    actual_median = np.median(results)
    actual_skew = stats.skew(results)
    actual_kurt = stats.kurtosis(results)

    if num_experiments >= 3:
        shapiro_stat, p_value = stats.shapiro(results)
    else:
        shapiro_stat, p_value = (None, None)

    # --- Main Page Layout ---
    st.title("🎲 Central Limit Theorem Dashboard")
    st.markdown(f"Simulating **{num_experiments}** experiments, each summing **{num_rolls}** rolls of a **{die_sides}-sided** die.")

    tab_main, tab_stats, tab_info = st.tabs(
        ["📊 Main Plot", "📈 In-Depth Analysis", "ℹ️ Project Info"]
    )

    with tab_main:
        st.header("Histogram vs. Theoretical Gaussian Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_width = max(1, int(sigma / 3))
        min_val = int(mu - 4*sigma)
        max_val = int(mu + 4*sigma)
        bins = np.arange(min_val, max_val + class_width, class_width)
        
        ax.hist(results, bins=bins, density=True, alpha=0.7,
                edgecolor='black', label=f'Histogram of {num_experiments} Results')
        
        x_curve = np.linspace(min_val, max_val, 300)
        y_curve = [gaussian_curve(val, mu, variance) for val in x_curve]
        ax.plot(x_curve, y_curve, 'r-', linewidth=2,
                label=f'Gaussian Curve (μ={mu:.2f}, σ²={variance:.2f})')
        
        ax.set_title("Distribution of Sums vs. Normal Distribution")
        ax.set_xlabel(f"Sum of {num_rolls} Dice Rolls")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
