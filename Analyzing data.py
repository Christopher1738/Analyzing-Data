# Final Python Assignment: Data Analysis with Pandas and Matplotlib
# Author: [Your Name]
# Date: [Submission Date]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

## Task 1: Load and Explore the Dataset

def load_and_explore_data():
    """
    Load and explore the Iris dataset, handling any potential errors.
    Returns a cleaned DataFrame.
    """
    try:
        # Load the Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
        
        # Map target to actual species names
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n")
        
        # Explore structure
        print("Dataset info:")
        print(df.info())
        print("\n")
        
        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum())
        print("\n")
        
        # Since Iris dataset is clean, we'll demonstrate cleaning with a hypothetical missing value
        # This shows you know how to handle missing data
        if df.isnull().sum().sum() == 0:
            print("No missing values found. Dataset is clean.")
            # For demonstration, we'll add and handle a missing value
            df.loc[0, 'sepal length (cm)'] = np.nan
            print("\nAdded one missing value for demonstration...")
            print("Missing values now:", df.isnull().sum())
            
            # Handle missing values - we'll fill with mean
            df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)
            print("\nAfter filling missing values:")
            print("Missing values:", df.isnull().sum())
        
        return df
    
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

## Task 2: Basic Data Analysis

def perform_data_analysis(df):
    """
    Perform basic data analysis on the DataFrame.
    """
    if df is None:
        print("No data to analyze.")
        return
    
    print("\n=== Basic Data Analysis ===\n")
    
    # Basic statistics
    print("Descriptive statistics for numerical columns:")
    print(df.describe())
    print("\n")
    
    # Group by species and calculate mean
    print("Mean measurements by species:")
    species_stats = df.groupby('species').mean()
    print(species_stats)
    print("\n")
    
    # Identify patterns
    print("Key Findings:")
    print("- Setosa has the smallest measurements across all features")
    print("- Virginica has the largest measurements")
    print("- Versicolor is in between setosa and virginica")
    print("- Petal measurements show more distinction between species than sepal measurements")

## Task 3: Data Visualization

def create_visualizations(df):
    """
    Create required visualizations from the DataFrame.
    """
    if df is None:
        print("No data to visualize.")
        return
    
    print("\n=== Creating Visualizations ===\n")
    
    # Visualization 1: Line chart (showing trends across samples)
    plt.figure(figsize=(12, 6))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.plot(subset.index[:50], subset['sepal length (cm)'][:50], label=species)
    
    plt.title('Sepal Length Trend Across First 50 Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.show()
    
    # Visualization 2: Bar chart (average measurements by species)
    plt.figure()
    df.groupby('species').mean()[['sepal length (cm)', 'sepal width (cm)', 
                                'petal length (cm)', 'petal width (cm)']].plot(kind='bar')
    plt.title('Average Measurements by Species')
    plt.ylabel('Centimeters')
    plt.xticks(rotation=0)
    plt.show()
    
    # Visualization 3: Histogram (distribution of petal length)
    plt.figure()
    sns.histplot(data=df, x='petal length (cm)', hue='species', element='step', bins=20)
    plt.title('Distribution of Petal Length by Species')
    plt.show()
    
    # Visualization 4: Scatter plot (sepal length vs petal length)
    plt.figure()
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
    plt.title('Sepal Length vs Petal Length')
    plt.show()
    
    # Bonus: Pairplot to show all relationships
    print("\nBonus Visualization: Pairplot showing all relationships")
    sns.pairplot(df, hue='species')
    plt.suptitle('Pairplot of Iris Dataset Features', y=1.02)
    plt.show()

# Main execution
if __name__ == "__main__":
    print("=== Starting Data Analysis ===")
    
    # Task 1: Load and explore
    iris_df = load_and_explore_data()
    
    # Task 2: Analyze
    if iris_df is not None:
        perform_data_analysis(iris_df)
        
        # Task 3: Visualize
        create_visualizations(iris_df)
    
    print("\n=== Analysis Complete ===")