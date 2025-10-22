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
    Simulates one experiment:
    Rolls a die `num_rolls` times and returns the sum.
    """
    return sum(random.randint(1, die_sides) for _ in range(num_rolls))

def calculate_theoretical_stats(num_rolls, die_sides):
    """Calculates the theoretical mean and variance."""
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
st.sidebar.title("MINT Projekt 2")
st.sidebar.header("Simulation Controls")

# INTERACTIVE WIDGET for the "Additional Question"
die_sides = st.sidebar.selectbox(
    "1. Die Type (Sides)",
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
