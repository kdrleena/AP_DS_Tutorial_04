# -*- coding: utf-8 -*-
"""
IA2 - AP_DS - 2025/2026 - Tutorial 04
Student Name: [Your Name]
Solutions to Exercises on Essential Python Libraries.
"""

# Import libraries at the top
import numpy as np
import pandas as pd

print("=" * 50)
print("EXERCISE 01: Temperature Analysis with NumPy")
print("=" * 50)

# --- Your Exercise 01 code will go here ---

print("\n" + "=" * 50)
print("EXERCISE 02: Hospital Data with Pandas")
print("=" * 50)

# --- Your Exercise 02 code will go here ---

print("\nAll tasks completed successfully!")
 
 
 
 
 
 # Task 1: Create the NumPy array
T = np.array([
    [22, 25, 20],  # Day 1
    [24, 27, 21],  # Day 2
    [19, 20, 22],  # Day 3
    [25, 29, 28],  # Day 4
    [26, 30, 27],  # Day 5
    [21, 21, 23],  # Day 6
    [20, 26, 25]   # Day 7
])
# Cities: Alger (Column 0), Annaba (Column 1), Oran (Column 2)
print("Temperature Array:\n", T)




# Task 2: Compute and display max temperatures
max_per_city = np.max(T, axis=0) # Max down each column
max_per_day = np.max(T, axis=1)  # Max across each row
overall_max = np.max(T)          # Max in the entire array

print(f"Max temperature per city (Alger, Annaba, Oran): {max_per_city}")
print(f"Max temperature per day: {max_per_day}")
print(f"Overall maximum temperature: {overall_max}")  


# Task 3: Find day and city of overall max
max_index_flat = np.argmax(T) # Gives a single number for the position in the flattened array
day_index, city_index = np.unravel_index(max_index_flat, T.shape) # Converts it to (row, column)



# Remember: Python indices start at 0, but our Days start at 1.
cities = ["Alger", "Annaba", "Oran"]
print(f"The overall max temperature was in {cities[city_index]} on Day {day_index + 1}.")



# Task 4: Higher temp between Alger (col 0) and Oran (col 2) for each day
higher_alger_oran = np.maximum(T[:, 0], T[:, 2])
print(f"Higher temperature between Alger and Oran for each day: {higher_alger_oran}")




# Task 5: Add a new row for the average of each city
average_temps = np.mean(T, axis=0) # Calculate average for each column (city)
# To add as a row, we need to reshape it and then use vstack
T_with_avg = np.vstack([T, average_temps])
print("Original array with average row added:\n", T_with_avg)



# Task 6: Days where Annaba was the hottest
# This means: (Annaba's temp > Alger's temp) AND (Annaba's temp > Oran's temp)
annaba_hottest_days = (T[:, 1] > T[:, 0]) & (T[:, 1] > T[:, 2])
# np.where returns the indices where the condition is True. Add 1 for actual day number.
days_list = np.where(annaba_hottest_days)[0] + 1
print(f"Annaba was the hottest city on days: {days_list}")



#STEP2:
# Task 1: Create the DataFrame
data = {
    'City': ['Alger', 'Annaba', 'Oran', 'Alger', 'Oran', 'Annaba', 'Alger', 'Oran'],
    'Department': ['Cardiology', 'Neurology', 'Orthopedics', 'Cardiology', 'Neurology', 'Orthopedics', 'Cardiology', 'Neurology'],
    'Age': [45, 60, 30, 50, 40, 70, 35, 55],
    'DaysAdmitted': [5, 8, 3, 7, 4, 6, 2, 9],
    'DailyCost': [200, 300, 150, 220, 280, 160, 210, 290],
    'Satisfaction': [4.5, 3.8, 4.2, 4.0, 4.1, 3.5, 4.7, 3.9],
    'Readmitted': [0, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)
print("\nDataFrame Info:")
print(df.info())




# Task 2: Add new columns
df['TotalCost'] = df['DaysAdmitted'] * df['DailyCost']

# Define a function to classify age
def get_age_group(age):
    if age < 30:
        return "Young"
    elif age <= 50:
        return "Middle"
    else:
        return "Senior"

df['AgeGroup'] = df['Age'].apply(get_age_group)
print("\nDataFrame with TotalCost and AgeGroup:")
print(df[['City', 'Age', 'AgeGroup', 'DaysAdmitted', 'DailyCost', 'TotalCost']])




# Task 3: Add, Rename, and Delete Columns
df['Bonus'] = df['DailyCost'] * 0.1  # Add Bonus column
print("\nDataFrame with Bonus column:")
print(df[['DailyCost', 'Bonus']].head())

df = df.rename(columns={'DailyCost': 'DailyFee'}) # Rename DailyCost to DailyFee
print("\nAfter renaming DailyCost to DailyFee:")
print(df.columns)

df = df.drop(columns=['Bonus']) # Delete Bonus column
print("\nAfter deleting Bonus column:")
print(df.columns)



# Task 4: Data Transformation
# Note: The threshold is probably 6000, not 60000 (check your values!)
df['HighCost'] = np.where(df['TotalCost'] > 6000, 'Yes', 'No')

def get_satisfaction_level(score):
    if score >= 8:
        return "Excellent"
    elif score >= 5:
        return "Average"
    else:
        return "Poor"

df['SatisfactionLevel'] = df['Satisfaction'].apply(get_satisfaction_level)
print("\nDataFrame with HighCost and SatisfactionLevel:")
print(df[['TotalCost', 'HighCost', 'Satisfaction', 'SatisfactionLevel']])



# Task 7: Risk Score Calculation (doing this before Task 5 for sorting)
df['RiskScore'] = (df['DaysAdmitted'] * 0.4) + (df['Age'] * 0.3) + ((100 - df['Satisfaction'] * 10) * 0.3)

def get_risk_category(score):
    if score > 60:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

df['RiskCategory'] = df['RiskScore'].apply(get_risk_category)
print("\nDataFrame with RiskScore and RiskCategory:")
print(df[['City', 'DaysAdmitted', 'Age', 'Satisfaction', 'RiskScore', 'RiskCategory']])



# Task 5: Ranking and Sorting
df['CostRank'] = df['TotalCost'].rank(ascending=False, method='dense').astype(int)
# Sort by RiskScore (descending)
df_sorted = df.sort_values('RiskScore', ascending=False)
print("\nDataFrame Sorted by RiskScore (Descending):")
print(df_sorted[['City', 'TotalCost', 'CostRank', 'RiskScore', 'RiskCategory']])




# Task 6 & 8: City Summary and find city with highest avg RiskScore
city_summary = df.groupby('City').agg(
    Average_TotalCost=('TotalCost', 'mean'),
    Average_Satisfaction=('Satisfaction', 'mean'),
    Readmission_Rate=('Readmitted', 'mean') # Mean of 1/0 gives the percentage
).round(2)

# Add the average RiskScore to the summary for Task 8
city_avg_risk = df.groupby('City')['RiskScore'].mean().round(2)
city_summary['Average_RiskScore'] = city_avg_risk

print("\nCity Summary:")
print(city_summary)

# Find city with highest average RiskScore
highest_risk_city = city_summary['Average_RiskScore'].idxmax()
print(f"\nCity with the highest average RiskScore: {highest_risk_city}")



# Task 9: Export to CSV
# Use the main DataFrame for hospital_risk.csv
df_sorted.to_csv('hospital_risk.csv', index=False)
# Use the city_summary DataFrame for city_summary.csv
city_summary.to_csv('city_summary.csv')

print("\nData exported to 'hospital_risk.csv' and 'city_summary.csv'.")




print("\n" + "=" * 50)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 50)  


