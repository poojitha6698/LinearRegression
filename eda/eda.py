import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df):

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Charges Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['charges'], kde=True)
    plt.title("Distribution of Charges")
    plt.show()

    # BMI vs Charges
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='bmi', y='charges', data=df)
    plt.title("BMI vs Charges")
    plt.show()