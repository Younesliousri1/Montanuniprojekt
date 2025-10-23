import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy.stats as stats
import qrcode
from io import BytesIO

# --- 1. Page Configuration  ---
st.set_page_config(
    page_title="Central Limit Theorem Dashboard",
    page_icon="🎲",
    layout="wide"
)

# ---  PUBLIC APP URL  ---
# Find this in your Streamlit Cloud "Manage app" menu.
# It should look like: https://your-app-name.streamlit.app
APP_URL = "https://testpy-xhdwrcnufmqcjxeyfvl4qu.streamlit.app/" 


# --- Generate QR Code ---
def generate_qr_code(url):
    """Generates a QR code image from a URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
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

    # --- MODIFICATION: Added Creators List Here ---
    with col2:
        st.subheader("Creators")
        st.markdown("""
        - **Younes Liousri:** `younes.liousri@stud.unileoben.ac.at`
        - **Alexander Holzinger:** `alexander.holzinger@stud.unileoben.ac.at`
        - **Asim Spahić:** `asim.spahic@stud.unileoben.ac.at`
        - **Muhammad Waseem:** `muhammad.waseem@stud.unileoben.ac.at`
        - **Andreas Weixlbaumer:** `andreas.weixlbaumer@stud.unileoben.ac.at`
        """)

        st.subheader("Project Details")
        st.markdown("""
        - **Project:** MINT Projekt 2
        - **Topic:** Würfeln und der Zentrale Grenzwertsatz
        - **Controls:** Use the sidebar (on the next page) to change parameters.
        """)

# --- 3. MAIN DASHBOARD LOGIC ---
else:
    # --- Core Experiment & Stat Functions ---
    def run_experiment(num_rolls, die_sides):
        """
        Simulates one experiment:
        Rolls a dice `num_rolls` times and returns the sum.
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
        st.pyplot(fig)

        st.header("The 'Before' Picture: Distribution of a Single Die Roll")
        st.markdown("The CLT is powerful because it takes a *uniform* distribution (one die roll) and produces a *normal* distribution (a bell curve) when you sum many of them.")
        
        single_die_data = pd.DataFrame(
            {'Probability': [1/die_sides] * die_sides},
            index=range(1, die_sides + 1)
        )
        st.bar_chart(single_die_data)

    with tab_stats:
        st.header("In-Depth Statistical Analysis")
        
        st.subheader("Theoretical vs. Actual Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Theoretical Mean (μ)", f"{mu:.3f}")
            st.metric("Theoretical Variance (σ²)", f"{variance:.3f}")
        with col2:
            st.metric("Actual Mean (from data)", f"{actual_mean:.3f}")
            st.metric("Actual Variance (from data)", f"{actual_var:.3f}")

        st.subheader("Distribution Shape Statistics")
        cols = st.columns(3)
        cols[0].metric("Median", f"{actual_median:.3f}", help="The 50th percentile (middle value).")
        cols[1].metric("Skewness", f"{actual_skew:.3f}", help="Measures asymmetry (Normal=0).")
        cols[2].metric("Kurtosis (Fisher)", f"{actual_kurt:.3f}", help="Measures 'tailedness' (Normal=0).")

        st.subheader("Professional Normality Tests")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Shapiro-Wilk Test")
            if p_value is None:
                st.error("Test requires at least 3 experiments to run.")
            else:
                st.metric("p-value", f"{p_value:.4f}")
                if p_value > 0.05:
                    st.success("**Conclusion:** The data appears to be normally distributed. (p > 0.05)")
                else:
                    st.warning("**Conclusion:** The data does *not* appear to be normally distributed. (p <= 0.05)")
            
        with col2:
            st.subheader("Q-Q (Quantile-Quantile) Plot")
            fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
            stats.probplot(results, dist="norm", plot=ax_qq)
            ax_qq.set_title("Normal Q-Q Plot")
            ax_qq.set_xlabel("Theoretical Quantiles")
            ax_qq.set_ylabel("Sample Quantiles")
            st.pyplot(fig_qq)
            st.markdown("If the data is normal, the blue dots should lie on the red line.")
            
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
        * **Task:** Roll a 6-sided die 20 times and sum the results.
        * **Group 1 ($G_1$):** Repeat this experiment 21 times.
        * **Group 2 ($G_2$):** Repeat this experiment 42 times.
        * **Goal:** Compare the resulting histogram to the specific Gaussian curve:
            $$f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^{2}}} \\exp\\left(-\\frac{(x-70)^{2}}{2\\sigma^{2}}\\right)$$
            ...where $\\mu = 70$ and $\\sigma^{2} = 20 \cdot \\frac{35}{12}$.
            
        """)
