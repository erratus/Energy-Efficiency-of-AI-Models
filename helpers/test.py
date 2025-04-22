import matplotlib.pyplot as plt

# Data
models = ["BERT", "CNN", "DistilBERT", "GPT-2", "ViT", "XGBoost", "GLaM"]
accuracy = [60.74, 72.12, 53.36, 55.84, 93.88, 61.31, 74.76]  # Percentage

# Style configuration
plt.style.use('default')
plt.rcParams.update({
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.titlepad': 20
})

# Create figure
plt.figure(figsize=(12, 6))

# Custom color gradient (red to green)
colors = plt.cm.RdYlGn([x/100 for x in accuracy])

# Plot bars
bars = plt.bar(models, accuracy, color=colors, edgecolor='black', linewidth=0.7, alpha=0.9)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height-5,
             f'{height:.1f}%',
             ha='center', va='top',
             color='white' if height > 70 else 'black',
             fontweight='bold')

# Customize plot
plt.title("Model Accuracy Comparison", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Models", fontsize=14, labelpad=10)
plt.ylabel("Accuracy (%)", fontsize=14, labelpad=10)
plt.ylim(0, 105)
plt.xticks(rotation=45)

# Add horizontal line at average
avg_accuracy = sum(accuracy)/len(accuracy)
plt.axhline(avg_accuracy, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
plt.text(len(models)-0.5, avg_accuracy+2, 
         f'Average: {avg_accuracy:.1f}%', 
         ha='right', color='red', fontsize=11)

# Highlight best performer
best_idx = accuracy.index(max(accuracy))
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(2)

plt.tight_layout()
plt.show()