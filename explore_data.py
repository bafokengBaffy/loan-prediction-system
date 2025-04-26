import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os

def main():
    # Set paths for the dataset and output reports folder
    data_path = "data/loan_data_set.csv"
    reports_dir = "reports"
    
    # Check for the reports directory and create it if it doesn't exist
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created '{reports_dir}' folder for outputs.")
    
    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found. Please make sure 'loan_data_set.csv' is in the 'data' folder.")
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")

    # Clean column names and data
    df.columns = df.columns.str.replace(r'[\/, ]', '_', regex=True)
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    
    # ============================
    # Task 1a: Missing values
    # ============================
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    missing_values_path = os.path.join(reports_dir, "missing_values_heatmap.png")
    plt.savefig(missing_values_path)
    plt.close()
    print(f"Missing values heatmap saved as {missing_values_path}")
    
    # ============================
    # Task 1b: Outliers (Boxplots)
    # ============================
    numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Detection: {col}")
        plt.tight_layout()
        outlier_path = os.path.join(reports_dir, f"outliers_{col}.png")
        plt.savefig(outlier_path)
        plt.close()
        print(f"Boxplot for {col} (outlier detection) saved as {outlier_path}")
    
    # ============================
    # Task 1c: Descriptive Analysis
    # ============================
    desc_stats = df.describe(include='all')
    desc_stats_path = os.path.join(reports_dir, "descriptive_statistics.csv")
    desc_stats.to_csv(desc_stats_path)
    print(f"Descriptive statistics saved to '{desc_stats_path}'")
    
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    correlation_heatmap_path = os.path.join(reports_dir, "correlation_heatmap.png")
    plt.savefig(correlation_heatmap_path)
    plt.close()
    print(f"Correlation heatmap saved as {correlation_heatmap_path}")
    
    # ============================
    # Task 8a: 4 Required Figures
    # ============================
    # 1. Gender Distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x='Gender')
    plt.title("Gender Distribution")
    plt.tight_layout()
    fig1_path = os.path.join(reports_dir, "fig1_gender_distribution.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"Gender Distribution plot saved as {fig1_path}")
    
    # 2. Loan Status Count
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x='Loan_Status')
    plt.title("Loan Status")
    plt.tight_layout()
    fig2_path = os.path.join(reports_dir, "fig2_loan_status.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"Loan Status plot saved as {fig2_path}")
    
    # 3. Education vs Loan Status
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x='Education', hue='Loan_Status')
    plt.title("Education vs Loan Status")
    plt.tight_layout()
    fig3_path = os.path.join(reports_dir, "fig3_education_vs_loanstatus.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"Education vs Loan Status plot saved as {fig3_path}")
    
    # 4. Property Area vs Loan Status
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x='Property_Area', hue='Loan_Status')
    plt.title("Property Area vs Loan Status")
    plt.tight_layout()
    fig4_path = os.path.join(reports_dir, "fig4_property_vs_loanstatus.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"Property Area vs Loan Status plot saved as {fig4_path}")
    
    print("\nAll visualizations and reports have been successfully saved in the 'reports/' folder.")

if __name__ == "__main__":
    main()
