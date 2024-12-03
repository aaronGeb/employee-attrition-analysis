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

    def plot_count_with_percentage(self, column: str) -> None:
        """
        Creates a horizontal count plot with percentage values displayed on each bar.

        :param column: The column name for which the count plot is generated.
        """
        if self.data is None or column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")

        count_values = self.data[column].value_counts()
        order = count_values.index

        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y=column, data=self.data, order=order)

        total = len(self.data)
        for p in ax.patches:
            width = p.get_width()
            percentage = (width / total) * 100
            ax.text(
                width + 2,
                p.get_y() + p.get_height() / 2,
                f"{percentage:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
            )

        plt.title(f"Distribution of {column.title()} in Percentage", fontsize=14)
        plt.xlabel("Count")
        plt.ylabel(column.title())
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self) -> None:
        """
        Plot a correlation matrix heatmap with values rounded to two decimal places.
        """
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Use 'load_data' or set 'data' attribute."
            )

        numeric_data = self.data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in the dataset.")

        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            linewidths=0.5,
        )
        plt.title("Correlation Matrix", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, x: str, y: str) -> None:
        """
        Create a scatter plot of two numerical columns.

        :param x: Name of the column to plot on the x-axis.
        :param y: Name of the column to plot on the y-axis.
        """
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Use 'load_data' or set 'data' attribute."
            )
        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError(f"Columns '{x}' or '{y}' not found in the dataset.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, data=self.data)
        plt.title(f"{x.title()} vs {y.title()}", fontsize=16)
        plt.xlabel(x.title())
        plt.ylabel(y.title())
        plt.tight_layout()
        plt.show()

    def plot_attrition_by_category(
        self, x_col: str, hue_col: str, sorted_order: bool = True
    ) -> None:
        """
        Plots a horizontal countplot of attrition by a specified category with percentage labels,
        optionally sorted by total count.

        Parameters:
        - x_col (str): The column for the x-axis (e.g., 'jobrole', 'department').
        - hue_col (str): The column for hue, typically the 'attrition' status.
        - sorted_order (bool): Whether to sort the categories by total count (default is True).
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Use 'load_data' to set the data.")

        if x_col not in self.data.columns or hue_col not in self.data.columns:
            raise ValueError(f"Columns '{x_col}' or '{hue_col}' not found in the data.")

        # Calculate total counts for sorting
        if sorted_order:
            sorted_categories = self.data[x_col].value_counts().index
        else:
            sorted_categories = self.data[x_col].unique()

        plt.figure(figsize=(12, 8))

        # Create horizontal countplot with sorted categories
        ax = sns.countplot(
            y=x_col, hue=hue_col, data=self.data, order=sorted_categories
        )
        plt.title(f"Attrition by {x_col.capitalize()} (Sorted)", fontsize=16)

        # Add percentage labels to the bars
        for p in ax.patches:
            width = p.get_width()

            # Calculate total count for the specific category and hue
            category = p.get_y() + p.get_height() / 2
            total = sum(
                self.data[self.data[x_col] == sorted_categories[category]].shape[0]
            )
            percentage = 100 * width / total

            # Annotate the bar with percentage
            ax.text(
                width + 2,
                p.get_y() + p.get_height() / 2,
                f"{percentage:.1f}%",
                va="center",
            )

            plt.show()

    def summary_statistics(self):
        return self.data.describe()

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
        plt.title(f"{column.title()} in Percentage")
        plt.show()

    def plot_attrition_by_jobrole(self, x_col: str, hue_col="attrition"):
        """
        Plots a horizontal countplot of attrition by job role with percentage labels,
        sorted by total count.

        Parameters:
        - data (pd.DataFrame): The dataset containing job roles and attrition columns.
        - x_col (str): The column for the x-axis (default is 'jobrole').
        - hue_col (str): The column for hue, typically the 'attrition' status (default is 'attrition').
        """
        # Ensure the x_col is categorical
        if not pd.api.types.is_categorical_dtype(self.data[x_col]):
            self.data[x_col] = self.data[x_col].astype("category")

        # Calculate total counts for sorting
        sorted_jobroles = self.data[x_col].value_counts().index

        # Set figure size
        plt.figure(figsize=(12, 8))

        # Create horizontal countplot with sorted x categories
        plot = sns.countplot(
            y=x_col, hue=hue_col, data=self.data, order=sorted_jobroles
        )
        plt.title(f"Attrition by {x_col.title()}")

        # Calculate total counts for each x category
        total_counts = self.data.groupby([x_col])[hue_col].count().to_dict()

        # Add percentage labels
        for p in plot.patches:
            # Extract bar values
            height = p.get_width()
            y_position = p.get_y() + p.get_height() / 2

            # Get the name of the y category (job role)
            job_role = p.get_y() + p.get_height() / 2
            category_index = int(round(p.get_y()))
            job_role = sorted_jobroles[category_index]

            # Calculate percentage
            total = total_counts[job_role]
            percentage = f"{(height / total) * 100:.1f}%"

            # Annotate percentage next to bars
            plot.annotate(
                percentage,
                (height, y_position),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        plt.show()

    def plot_age_distribution_by_gender_and_attrition(
        self, x_col: str, y_col: str, hue_col="attrition"
    ):
        """
        Plots a violin plot of age distribution by gender with attrition status overlay.

        Parameters:
        data (DataFrame): The DataFrame containing 'age', 'gender', and 'attrition' columns.
        """
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=self.data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            split=True,
            palette="coolwarm",
        )

        plt.title("Age Distribution Among Genders and Attrition Status")
        plt.xlabel("Gender")
        plt.ylabel("Age")
        plt.grid(axis="y", alpha=0.75)
        plt.legend(title="Attrition", loc="upper left")
        plt.show()

    def plot_attrition_by_category(data, category_column, attrition_column="attrition"):
        """
        Plots a stacked bar plot showing the proportion of attrition status across different categories.

        Parameters:
        - data: DataFrame containing the data.
        - category_column: The column representing the categorical variable (e.g., 'department' or 'job role').
        - attrition_column: The column representing the attrition status ('yes'/'no').
        """
        # Count the occurrences of attrition status in each category (e.g., department or job role)
        attrition_counts = (
            data.groupby([category_column, attrition_column])
            .size()
            .unstack(fill_value=0)
        )

        # Plot stacked bar plot
        attrition_counts.plot(
            kind="bar", stacked=True, color=["red", "green"], figsize=(8, 6)
        )

        # Customize plot
        plt.title(f"Attrition Status Across Different {category_column.capitalize()}")
        plt.xlabel(category_column.capitalize())
        plt.ylabel("Count of Individuals")
        plt.xticks(rotation=0)
        plt.legend(title="Attrition Status", labels=["No", "Yes"])

        # Show plot
        plt.tight_layout()
        plt.show()

    def plot_pairplot_by_attrition(
        self, hue_column="attrition", palette="coolwarm", diag_kind="kde", height=3
    ):
        """
        Plots a pairplot of numerical features, colored by attrition status.

        Parameters:
        - data: DataFrame containing the data.
        - hue_column: The column to use for coloring the data points (default is 'Attrition').
        - palette: Color palette for the plot (default is 'coolwarm').
        - diag_kind: Type of plot for the diagonal axes (default is 'kde').
        - height: Height of each facet (default is 3).
        """
        # Create the pairplot
        # Select relevant columns for the pairplot
        selected_columns = ["age", "yearsatcompany", "monthlyincome", hue_column]
        data_selected = self.data[selected_columns]

        # Create the pairplot
        sns.pairplot(
            data_selected,
            hue=hue_column,
            palette=palette,
            diag_kind=diag_kind,
            height=height,
        )

        # Customize plot title
        plt.suptitle("Pair Plot of Numerical Features by Attrition", y=1.02)

        # Show the plot
        plt.show()

    def plot_salary_distribution_by_attrition_and_gender(
        self, hue_column:str, palette="Set2"
    ):
        """
        Plots a boxplot showing salary distribution by attrition status and gender.

        Parameters:
        - data: DataFrame containing the data.
        - hue_column: The column to use for coloring the data points (default is 'Gender').
        - palette: Color palette for the plot (default is 'Set2').
        """
        # Create the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            x="attrition", y="monthlyincome", hue=hue_column, data=self.data, palette=palette
        )

        # Customize plot title and labels
        plt.title("Monthly Income Distribution by Attrition and Gender")
        plt.xlabel("Attrition")
        plt.ylabel("MonthlyIncome")

        # Show the plot
        plt.show()
