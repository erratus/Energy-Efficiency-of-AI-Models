import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["BERT", "CNN", "DistilBERT", "GPT-2", "ViT", "XGBoost", "GLaM"]
energy_consumed_joules = [15921.49, 3000.64, 16591.18, 33751.45, 351345.45, 786.73, 1938.10]
carbon_intensity = 475  # gCO2/kWh (global average)

# Conversion function
def calculate_carbon_footprint(energy_joules):
    return (energy_joules / 3.6e6) * carbon_intensity

carbon_footprints = [calculate_carbon_footprint(e) for e in energy_consumed_joules]

# Create figure with log scale
plt.figure(figsize=(12, 6))
bars = plt.bar(models, carbon_footprints, color='#1f77b4', edgecolor='black', alpha=0.8)

# Configure log scale
plt.yscale('log')
plt.ylim(1, 1000)  # Set bounds to include all values

# Custom grid lines
plt.yticks([1, 10, 100, 1000], ["1g", "10g", "100g", "1kg"])
plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)

# Annotate exact values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height*1.2,  # Position above bar
             f'{height:.1f}g', 
             ha='center', va='bottom',
             fontsize=10)

# Labels and title
plt.title("Carbon Footprint of Model Training (Log Scale)", 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel("Model", fontsize=14)
plt.ylabel("CO₂ Emissions (log scale)", fontsize=14)
plt.xticks(rotation=45)

# Add contextual references
plt.axhline(10, color='grey', linestyle=':', alpha=0.5)
plt.axhline(100, color='grey', linestyle=':', alpha=0.5)
plt.text(len(models)-0.5, 12, "≈1h laptop use", ha='right', color='grey', fontsize=9)
plt.text(len(models)-0.5, 120, "≈5km car travel", ha='right', color='grey', fontsize=9)

plt.tight_layout()
plt.show()