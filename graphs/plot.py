import matplotlib.pyplot as plt

# Data
categories = ['1 person', '2 persons', '3 persons', '4 persons', '5 persons', '6 persons', 'GPT-3']
kwh = [611, 891, 948, 1112, 1211, 1202, 1287000]

# Create plot
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, kwh, color='skyblue')

# Add value labels on each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,} kWh',
             ha='center', va='bottom',
             fontsize=9)

# Log scale and labels
plt.yscale('log')
plt.ylabel("Energy Consumption (kWh, log scale)")
plt.title("Energy Consumption Comparison")

# Rotate x-labels
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()