import matplotlib.pyplot as plt

# Data
countries = ['China', 'Germany', 'Japan', 'Mexico', 'Canada', 'Saudi Arabia',
             'Ireland', 'Italy', 'South Korea', 'Vietnam', 'India', 'Rest of World']
values = [343, 74, 67, 54, 35, 28, 26, 25, 25, 25, 24, 1]
colors = ['red', 'gold', 'skyblue', 'green', 'darkred', 'brown',
          'lime', 'orange', 'blue', 'purple', 'darkolivegreen', 'lightgrey']

# Plot
plt.figure(figsize=(8, 8))
wedges, _ = plt.pie(values, colors=colors, startangle=280, counterclock=False)

# Legend
legend_labels = [f"{country} ${value}B {round(value * 100/sum(values), 1)}%" for country, value in zip(countries, values)]
plt.legend(wedges, legend_labels, title="Countries", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Title
plt.title('US Balance of Trade Deficit (2014)')

# Show plot
plt.show()