import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from pandas.plotting import parallel_coordinates

# Data
models = ["BERT", "CNN", "DistilBERT", "GPT-2", "ViT", "XGBoost", "GLaM"]
energy_consumed = [15921.49, 3000.64, 16591.18, 33751.45, 351345.45, 786.73, 1938.10]  # Joules
accuracy = [60.74, 72.12, 53.36, 55.84, 93.88, 61.31, 74.76]  # Percentage
training_time = [767.12, 329.52, 703.70, 1443.05, 5019.22, 35.70, 205.48]  # Seconds

# Style configuration
plt.style.use('seaborn')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['font.size'] = 12

# 1. Energy Consumption vs. Accuracy
plt.figure(figsize=(12, 7))
plt.scatter(energy_consumed, accuracy, color='royalblue', s=200, edgecolor='black', alpha=0.8)
plt.title("Energy Consumption vs. Accuracy", fontsize=18, pad=20, fontweight='bold')
plt.xlabel("Energy Consumed (Joules)", fontsize=14, labelpad=10)
plt.ylabel("Accuracy (%)", fontsize=14, labelpad=10)
plt.xscale('log')

for i, model in enumerate(models):
    plt.annotate(model, (energy_consumed[i], accuracy[i]), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# 2. Training Time Comparison
plt.figure(figsize=(12, 7))
bars = plt.bar(models, training_time, color='mediumseagreen', 
              edgecolor='black', alpha=0.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height+50,
             f'{height:.1f}s', ha='center', va='bottom', fontsize=11)

plt.title("Training Time Comparison", fontsize=18, pad=20, fontweight='bold')
plt.xlabel("Model", fontsize=14, labelpad=10)
plt.ylabel("Training Time (Seconds)", fontsize=14, labelpad=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Training Time vs. Accuracy
plt.figure(figsize=(12, 7))
plt.scatter(training_time, accuracy, color='purple', s=200, edgecolor='black', alpha=0.8)
plt.title("Training Time vs. Accuracy", fontsize=18, pad=20, fontweight='bold')
plt.xlabel("Training Time (Seconds)", fontsize=14, labelpad=10)
plt.ylabel("Accuracy (%)", fontsize=14, labelpad=10)

for i, model in enumerate(models):
    plt.annotate(model, (training_time[i], accuracy[i]), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# 4. Energy Efficiency (Accuracy per Joule)
efficiency = [acc/energy for acc, energy in zip(accuracy, energy_consumed)]

plt.figure(figsize=(12, 7))
bars = plt.bar(models, efficiency, color='darkorange', 
              edgecolor='black', alpha=0.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height+0.0001,
             f'{height*1000:.2f}x10⁻³', ha='center', va='bottom', fontsize=11)

plt.title("Energy Efficiency (Accuracy per Joule)", fontsize=18, pad=20, fontweight='bold')
plt.xlabel("Model", fontsize=14, labelpad=10)
plt.ylabel("Accuracy / Joule (×10⁻³)", fontsize=14, labelpad=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Radar Chart Comparison
def normalize(data):
    return [x/max(data) for x in data]

norm_acc = normalize(accuracy)
norm_speed = normalize([1/t for t in training_time])
norm_eff = normalize(efficiency)

categories = ['Accuracy', 'Speed (1/Time)', 'Energy Efficiency']
N = len(categories)
angles = [n/float(N)*2*pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=12, color='navy')

for i, model in enumerate(models):
    values = [norm_acc[i], norm_speed[i], norm_eff[i]]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.1)

plt.title("Model Comparison Radar Chart", fontsize=18, pad=40, fontweight='bold')
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
plt.tight_layout()
plt.show()

# 6. Energy-Time Bubble Chart (Size = Accuracy)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(energy_consumed, training_time, 
                     s=[a*20 for a in accuracy], 
                     c=accuracy, cmap='viridis', 
                     edgecolor='black', alpha=0.8)

plt.title("Energy vs Training Time\n(Bubble Size = Accuracy)", 
          fontsize=18, pad=20, fontweight='bold')
plt.xlabel("Energy Consumed (Joules)", fontsize=14, labelpad=10)
plt.ylabel("Training Time (Seconds)", fontsize=14, labelpad=10)
plt.xscale('log')
plt.yscale('log')

cbar = plt.colorbar(scatter)
cbar.set_label('Accuracy (%)', fontsize=12)

for i, model in enumerate(models):
    plt.annotate(model, (energy_consumed[i], training_time[i]), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# 7. Parallel Coordinates Plot
data = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Training Time': training_time,
    'Energy': energy_consumed
})

for col in data.columns[1:]:
    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

plt.figure(figsize=(12, 7))
parallel_coordinates(data, 'Model', colormap='tab10', alpha=0.8, linewidth=2)
plt.title("Parallel Coordinates Comparison", fontsize=18, pad=20, fontweight='bold')
plt.xlabel("Metrics", fontsize=14, labelpad=10)
plt.ylabel("Normalized Value", fontsize=14, labelpad=10)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()