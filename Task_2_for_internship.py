import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visual style for clean modern plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
})

def load_and_clean_data(file_path):
    # Load CSV without parse_dates first to inspect columns
    df = pd.read_csv(file_path, dayfirst=True)
    
    # Print the columns to debug missing or misnamed columns
    print("Columns in the dataset:", df.columns.tolist())

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Verify 'Date' column exists now, else raise error with informative message
    if 'Date' not in df.columns:
        raise ValueError("The dataset does NOT contain a 'Date' column after stripping whitespace.")
    
    # Now parse 'Date' column as datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Drop rows where 'Date' could not be parsed
    df.dropna(subset=['Date'], inplace=True)

    # Strip whitespace in 'Area' and 'Region' columns if present
    if 'Area' in df.columns:
        df['Area'] = df['Area'].str.strip()
    if 'Region' in df.columns:
        df['Region'] = df['Region'].str.strip()

    # Drop rows missing essential columns
    df.dropna(subset=['Estimated Unemployment Rate (%)', 'Region'], inplace=True)
    
    # Filter only monthly frequency dataset
    df = df[df['Frequency'] == 'Monthly']
    
    # Ensure unemployment rate is numeric
    df['Estimated Unemployment Rate (%)'] = pd.to_numeric(df['Estimated Unemployment Rate (%)'], errors='coerce')
    df.dropna(subset=['Estimated Unemployment Rate (%)'], inplace=True)
    
    # Sort data by date for each region/area for orderly analysis
    df.sort_values(by=['Region', 'Area', 'Date'], inplace=True)
    
    return df

def plot_unemployment_trends(df):
    plt.figure(figsize=(16,10))
    sns.lineplot(
        data=df, x='Date', y='Estimated Unemployment Rate (%)',
        hue='Region', style='Area', markers=True, dashes=False)
    plt.title("Unemployment Rate Trends in India by Region and Area (2019-2020)")
    plt.xlabel("Date")
    plt.ylabel("Estimated Unemployment Rate (%)")
    plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', alpha=0.7, label='Covid-19 Lockdown Start (Mar 2020)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def covid_impact_analysis(df):
    cutoff_date = pd.to_datetime('2020-03-01')
    before_covid = df[df['Date'] < cutoff_date]
    after_covid = df[df['Date'] >= cutoff_date]
    
    mean_before = before_covid.groupby(['Region', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()
    mean_after = after_covid.groupby(['Region', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    covid_compare = pd.merge(
        mean_before, mean_after,
        on=['Region', 'Area'],
        suffixes=('_Before_Covid', '_After_Covid')
    )
    covid_compare['Increase (%)'] = covid_compare['Estimated Unemployment Rate (%)_After_Covid'] - covid_compare['Estimated Unemployment Rate (%)_Before_Covid']
    
    print("\nAverage Unemployment Rate Before vs After Covid-19 Lockdown (Mar 2020):")
    print(covid_compare.sort_values('Increase (%)', ascending=False))
    
    covid_compare_sorted = covid_compare.sort_values('Increase (%)', ascending=False)
    plt.figure(figsize=(14,8))
    sns.barplot(data=covid_compare_sorted, x='Increase (%)', y='Region', hue='Area')
    plt.title('Increase in Average Unemployment Rate After Covid-19 Lockdown by Region and Area')
    plt.xlabel('Increase in Unemployment Rate (%)')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.show()

def seasonal_trend_analysis(df):
    df['Month'] = df['Date'].dt.month
    
    monthly_avg = df.groupby(['Month', 'Region', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    sample_regions = ['Uttar Pradesh', 'Bihar', 'Delhi', 'Maharashtra', 'Karnataka'] 
    plt.figure(figsize=(16,10))
    sns.lineplot(
        data=monthly_avg[monthly_avg['Region'].isin(sample_regions)],
        x='Month', y='Estimated Unemployment Rate (%)',
        hue='Region', style='Area', markers=True, dashes=False)
    plt.title('Average Monthly Unemployment Rate by Region and Area (Seasonal Trends)')
    plt.xlabel('Month')
    plt.ylabel('Unemployment Rate (%)')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Region / Area', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    data_path = "Unemployment in India.csv"
    df = load_and_clean_data(data_path)

    print(f"Data Loaded: {len(df)} records")
    print("Unique Regions:", df['Region'].nunique())
    print("Date range:", df['Date'].min().date(), "to", df['Date'].max().date())
    
    plot_unemployment_trends(df)
    covid_impact_analysis(df)
    seasonal_trend_analysis(df)

    print(
        "\nInsights:\n"
        "1. Significant spike in unemployment starting April 2020 due to Covid-19 lockdown across regions and area types.\n"
        "2. Urban areas tend to have higher spikes and volatility compared to rural counterparts.\n"
        "3. Seasonal trends observed: generally higher unemployment in April-June and lower in Nov-Jan for select regions.\n"
        "4. Highest Covid impact noted in states like Bihar, Jharkhand, Haryana, and Tripura.\n"
        "5. Policies should emphasize urban job support and pandemic resilience measures.\n"
        "6. Diversifying employment and strengthening labor markets are critical for future shock mitigation."
    )

if __name__ == "__main__":
    main()
