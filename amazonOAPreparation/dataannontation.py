import matplotlib.pyplot as plt
# Data
countries = ['China', 'Germany', 'Japan', 'Mexico', 'Canada', 'Saudi Arabia',
             'Ireland', 'Italy', 'South Korea', 'Vietnam', 'India', 'Rest of World']
values = [343, 74, 67, 54, 35, 28, 26, 25, 25, 25, 24, 1]
percentages = [47.2, 10.2, 9.2, 7.4, 4.9, 3.9, 3.6, 3.5, 3.4, 3.4, 3.3, 0.1]
colors = ['red', 'gold', 'lightskyblue', 'limegreen', 'darkred', 'olive',
          'purple', 'blue', 'darkblue', 'orange', 'yellowgreen', 'lightgray']

# Combine country names with their corresponding values
legend_labels = [f"{country} ${value} {percent}%" for country, value, percent in zip(countries, values, percentages)]

# Create pie chart
fig, ax = plt.subplots(figsize=(10, 7))
wedges, _, autotexts = ax.pie(values, colors=colors, startangle=90, counterclock=False, autopct='%1.1f%%')

# Add legend
ax.legend(wedges, legend_labels, title="Countries", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Title
plt.title("US Balance of Trade Deficit (2014)")

# Display the plot
plt.show()