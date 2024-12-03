'''
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from typing import Optional


class Plotting:
    def __init__(self, data: Optional[DataFrame] = None):
        """
        Initialize the Plotting class.

        :param data: Optional initial DataFrame to work with. Defaults to None.
        """
        self.data = data

    def plot_count_with_percentage(
        self, column: str, title: str, xlabel: str, ylabel: str
    ) -> None:
        """
        Creates a horizontal count plot with percentage values displayed on each bar.

        Parameters:
        - data: pandas DataFrame containing the dataset
        - column: string, the column name for which the count plot is generated
        """
        # Get the count of each category in the specified column
        count_values = self.data[column].value_counts()

        # Sort the categories by count
        order = count_values.index

        # Create the count plot
        ax = sns.countplot(y=column, data=self.data, order=order)

        # Add percentage labels to the bars
        total = len(self.data)  # Total number of entries in the dataset

        # Loop through each bar and annotate it with the percentage value
        for p in ax.patches:
            width = p.get_width()  # Get the width (count) of the bar
            percentage = (width / total) * 100  # Calculate percentage
            ax.text(
                width + 2,
                p.get_y() + p.get_height() / 2,
                f"{percentage:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
            )

        # Add title and show the plot
        plt.title(f"{column} with Percentage")
        plt.show()

    def load_data(self, path: str) -> None:
        """
        Load data from a CSV file.

        :param path: Path to the CSV file.
        """
        try:
            self.data = pd.read_csv(path)
            print("Data loaded successfully.")
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

    def plot_attrition(self, column):
        """
        Creates a horizontal count plot with percentage values displayed on each bar.

        Parameters:
        - data: pandas DataFrame containing the dataset
        - column: string, the column name for which the count plot is generated
        """
        # Get the count of each category in the specified column
        count_values = self.data[column].value_counts()

        # Sort the categories by count
        order = count_values.index

        # Create the count plot
        ax = sns.countplot(y=column, data=self.data, order=order)

        # Add percentage labels to the bars
        total = len(self.data)  # Total number of entries in the dataset

        # Loop through each bar and annotate it with the percentage value
        for p in ax.patches:
            width = p.get_width()  # Get the width (count) of the bar
            percentage = (width / total) * 100  # Calculate percentage
            ax.text(
                width + 2,
                p.get_y() + p.get_height() / 2,
                f"{percentage:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
            )

        # Add title and show the plot
        plt.title(f"{column} with Percentage")
        plt.show()

    def summary_statistics(self):
        return self.data.describe()

    def plot_correlation_matrix(self) -> None:
        """
        Plot a correlation matrix heatmap with values rounded to two decimal places.

        :return: None
        """
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Use 'load_data' or set 'data' attribute."
            )

        # Select only numerical columns for correlation
        df = self.data.select_dtypes(include=[np.number])

        # Calculate the correlation matrix
        correlation_matrix = df.corr()

        # Plot the heatmap with annotations rounded to 2 decimal places
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",  # Format annotations to 2 decimal places
            cmap="coolwarm",  # Optional: use a visually appealing colormap
            cbar=True,  # Show color bar
            linewidths=0.5,  # Add spacing between cells
        )
        plt.title("Correlation Matrix", fontsize=16)
        plt.show()

    def plot_scatter(self, x: str, y: str):
        """
        Create a scatter plot of two numerical columns.

        :param x: Name of the column to plot on the x-axis.
        :param y: Name of the column to plot on the y-axis.
        """
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Use 'load_data' or set 'data' attribute."
            )

        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, data=self.data)
        plt.title(f"{x.capitalize()} vs {y.capitalize()}", fontsize=16)
        plt.xlabel(x.title())
        plt.ylabel(y.title())
        plt.show()

    def plot_attrition_by_category(self, x_col, hue_col="attrition", sorted_order=True):
        """
        Plots a horizontal countplot of attrition by a specified category with percentage labels,
        optionally sorted by total count.

        Parameters:
        - data (pd.DataFrame): The dataset containing the category and attrition columns.
        - x_col (str): The column for the x-axis (e.g., 'jobrole', 'department').
        - hue_col (str): The column for hue, typically the 'attrition' status (default is 'attrition').
        - sorted_order (bool): Whether to sort the categories by total count (default is True).
        """
        # Ensure the x_col is categorical
        if not pd.api.types.is_categorical_dtype(self.data[x_col]):
            self.data[x_col] = self.data[x_col].astype("category")

        # Calculate total counts for sorting
        if sorted_order:
            sorted_categories = self.data[x_col].value_counts().index
        else:
            sorted_categories = self.data[x_col].cat.categories

        # Set figure size
        plt.figure(figsize=(12, 8))

        # Create horizontal countplot with sorted categories
        plot = sns.countplot(y=x_col, hue=hue_col, data=self.data, order=sorted_categories)
        plt.title(f"Attrition by {x_col.capitalize()} (Sorted)")

        # Calculate total counts for each x category
        total_counts = self.data.groupby([x_col])[hue_col].count().to_dict()

        # Add percentage labels
        for p in plot.patches:
            # Extract bar values
            width = p.get_width()
            y_position = p.get_y() + p.get_height() / 2

            # Get the name of the y category (department or job role)
            category_index = int(round(p.get_y()))
            category_name = sorted_categories[category_index]

            # Calculate percentage
            total = total_counts[category_name]
            percentage = f"{(width / total) * 100:.1f}%"

            # Annotate percentage next to bars
            plot.annotate(
                percentage,
                (width, y_position),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        plt.show()
'''
