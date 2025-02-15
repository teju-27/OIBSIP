import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to load dataset
def load_dataset():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset Loaded Successfully!")
        display_data_info()

# Function to display dataset information
def display_data_info():
    if data is not None:
        info_text.set(f"Dataset Loaded: {len(data)} rows, {len(data.columns)} columns")
    else:
        messagebox.showerror("Error", "No dataset loaded!")

# Function to visualize unemployment trends
def plot_unemployment_trend():
    if data is not None and "Date" in data.columns and "Unemployment Rate" in data.columns:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=data['Date'], y=data['Unemployment Rate'], marker="o", color="b")
        plt.xlabel("Date")
        plt.ylabel("Unemployment Rate (%)")
        plt.title("Unemployment Rate Over Time")
        plt.xticks(rotation=45)
        plt.show()
    else:
        messagebox.showerror("Error", "Dataset is missing required columns!")

# Function to visualize unemployment by region
def plot_region_unemployment():
    if data is not None and "Region" in data.columns and "Unemployment Rate" in data.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=data['Region'], y=data['Unemployment Rate'], palette="viridis")
        plt.xlabel("Region")
        plt.ylabel("Unemployment Rate (%)")
        plt.title("Unemployment Rate by Region")
        plt.xticks(rotation=45)
        plt.show()
    else:
        messagebox.showerror("Error", "Dataset is missing required columns!")

# Function to show correlation heatmap
def plot_correlation_heatmap():
    if data is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded!")

# Initialize Tkinter window
root = tk.Tk()
root.title("Unemployment Analysis")

# Create UI Elements
info_text = tk.StringVar()
info_text.set("No dataset loaded")

tk.Label(root, text="Unemployment Analysis", font=("Arial", 16, "bold")).pack(pady=10)
tk.Label(root, textvariable=info_text, font=("Arial", 12)).pack()

tk.Button(root, text="Load Dataset", command=load_dataset, font=("Arial", 12), bg="lightblue").pack(pady=5)
tk.Button(root, text="Plot Unemployment Trend", command=plot_unemployment_trend, font=("Arial", 12)).pack(pady=5)
tk.Button(root, text="Plot Region Unemployment", command=plot_region_unemployment, font=("Arial", 12)).pack(pady=5)
tk.Button(root, text="Plot Correlation Heatmap", command=plot_correlation_heatmap, font=("Arial", 12)).pack(pady=5)

# Run the UI
root.mainloop()