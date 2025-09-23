#!/usr/bin/env python3
"""Demo script showing all regression types with dummy retail data.

Generates a 2x2 subplot showing linear, power, logarithmic, and exponential regression fits.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyretailscience.plots.styles import graph_utils as gu

# Set random seed for reproducible results
np.random.seed(42)

def generate_dummy_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate realistic retail dummy data for different regression types."""
    # Linear relationship: Sales vs Marketing Spend (with noise)
    marketing_spend = np.linspace(1000, 10000, 50)
    linear_sales = 2.5 * marketing_spend + 5000 + np.random.normal(0, 2000, 50)

    # Power law: Price vs Demand (price elasticity)
    prices = np.linspace(10, 100, 40)
    power_demand = 50000 * prices**(-1.2) + np.random.normal(0, 200, 40)
    power_demand = np.maximum(power_demand, 50)  # Ensure positive values

    # Logarithmic: Ad Spend vs Conversion Rate (diminishing returns)
    ad_spend = np.linspace(500, 20000, 45)
    log_conversions = 15 * np.log(ad_spend) - 80 + np.random.normal(0, 5, 45)
    log_conversions = np.maximum(log_conversions, 1)  # Ensure positive values

    # Exponential: Days vs Active Users (decay/retention)
    days = np.linspace(0, 30, 35)
    exp_users = 10000 * np.exp(-0.05 * days) + np.random.normal(0, 200, 35)
    exp_users = np.maximum(exp_users, 100)  # Ensure positive values

    return {
        "linear": (marketing_spend, linear_sales),
        "power": (prices, power_demand),
        "logarithmic": (ad_spend, log_conversions),
        "exponential": (days, exp_users),
    }

def create_regression_demo() -> plt.Figure:
    """Create a 2x2 subplot demonstrating all regression types."""
    # Generate data
    data = generate_dummy_data()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("PyRetailScience Regression Types Demo", fontsize=16, fontweight="bold")

    # Define plot configurations
    plot_configs = [
        {
            "ax": axes[0, 0],
            "data": data["linear"],
            "regression_type": "linear",
            "title": "Linear Regression\n(Sales vs Marketing Spend)",
            "xlabel": "Marketing Spend ($)",
            "ylabel": "Sales ($)",
            "color": "red",
        },
        {
            "ax": axes[0, 1],
            "data": data["power"],
            "regression_type": "power",
            "title": "Power Law Regression\n(Price Elasticity)",
            "xlabel": "Price ($)",
            "ylabel": "Demand (units)",
            "color": "blue",
        },
        {
            "ax": axes[1, 0],
            "data": data["logarithmic"],
            "regression_type": "logarithmic",
            "title": "Logarithmic Regression\n(Ad Spend Diminishing Returns)",
            "xlabel": "Ad Spend ($)",
            "ylabel": "Conversion Rate (%)",
            "color": "green",
        },
        {
            "ax": axes[1, 1],
            "data": data["exponential"],
            "regression_type": "exponential",
            "title": "Exponential Regression\n(User Retention Decay)",
            "xlabel": "Days",
            "ylabel": "Active Users",
            "color": "purple",
        },
    ]

    # Create each subplot
    for config in plot_configs:
        ax = config["ax"]
        x_data, y_data = config["data"]

        # Create scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=30, color="lightgray", label="Data")

        # Add regression line
        gu.add_regression_line(
            ax,
            regression_type=config["regression_type"],
            color=config["color"],
            linestyle="--",
            text_position=0.05,
            show_equation=True,
            show_r2=True,
        )

        # Customize plot
        ax.set_title(config["title"], fontweight="bold", pad=20)
        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the plot
    output_file = "regression_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Regression demo saved as: {output_file}")

    # Show the plot
    plt.show()

    return fig

if __name__ == "__main__":
    print("Generating PyRetailScience regression demo...")
    create_regression_demo()
    print("Demo complete! Check regression_demo.png for the output.")
