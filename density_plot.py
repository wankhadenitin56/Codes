
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the directory path for saving the plot
path = "D:\\Project Work\\MAIN NITIN PROJECT\\density plot\\3"
os.makedirs(path, exist_ok=True)

# Read the CSV file into a DataFrame
df = pd.read_csv("G:\\My Drive\\My project\\normaisation\\density.csv")

# Create a density plot using Seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df["target"] == "Inhibitors"]["ATS2e"], label="Inhibitors", fill=True, common_norm=False)
sns.kdeplot(data=df[df["target"] == "Non-inhibitors"]["ATS2e"], label="Non-inhibitors", fill=True, common_norm=False)

# Mark the cutpoint line (example: at x=0.192)
cutpoint = 0.200
plt.axvline(x=cutpoint, color='red', linestyle='--', label='Cutpoint')
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend(title="Class")

# Save the plot as an image file in the specified directory
plt.savefig(os.path.join(path, "density_plot.png"), dpi=300)

# Show the plot
plt.show()
