# ============================================================
#   SLEEP DISORDER DETECTION - EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'Arial'

COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
DISORDER_COLORS = {
    'None': '#2ecc71',
    'Insomnia': '#3498db',
    'Sleep Apnea': '#e74c3c',
    'Narcolepsy': '#9b59b6',
    'Restless Leg Syndrome': '#f39c12'
}

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel("sleep_disorder_merged_dataset.xlsx",
                   sheet_name="Sleep_Dataset", header=1)
df.columns = df.columns.str.replace('\n', '_').str.strip()
df = df[df['Patient_ID'] != 'Patient_ID'].reset_index(drop=True)
df['Sleep_Disorder_Type'] = df['Sleep_Disorder_Type'].fillna('None')

print("Data loaded successfully!")
print(f"Shape: {df.shape}")
print("Starting EDA...\n")


# ============================================================
# PLOT 1: CLASS DISTRIBUTION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PLOT 1 — Sleep Disorder Distribution', fontsize=14, fontweight='bold')

# Bar chart
disorder_counts = df['Sleep_Disorder_Type'].value_counts()
bars = axes[0].bar(disorder_counts.index, disorder_counts.values,
                   color=[DISORDER_COLORS[d] for d in disorder_counts.index],
                   edgecolor='white', linewidth=1.2)
axes[0].set_title('Count per Disorder Type', fontsize=12)
axes[0].set_xlabel('Disorder Type')
axes[0].set_ylabel('Number of Patients')
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, disorder_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=10, fontweight='bold')

# Pie chart
axes[1].pie(disorder_counts.values, labels=disorder_counts.index,
            colors=[DISORDER_COLORS[d] for d in disorder_counts.index],
            autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Percentage Distribution', fontsize=12)

plt.tight_layout()
plt.savefig('eda_01_class_distribution.png', bbox_inches='tight')
plt.show()
print("Plot 1 saved: eda_01_class_distribution.png")


# ============================================================
# PLOT 2: DEMOGRAPHICS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('PLOT 2 — Demographics Overview', fontsize=14, fontweight='bold')

# Age distribution by disorder
for disorder, color in DISORDER_COLORS.items():
    subset = df[df['Sleep_Disorder_Type'] == disorder]['Age']
    axes[0].hist(subset, bins=15, alpha=0.5, label=disorder, color=color)
axes[0].set_title('Age Distribution by Disorder')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=7)

# Gender split
gender_disorder = df.groupby(['Sleep_Disorder_Type', 'Gender']).size().unstack()
gender_disorder.plot(kind='bar', ax=axes[1], color=['#e91e8c', '#1565c0'],
                     edgecolor='white')
axes[1].set_title('Gender vs Disorder Type')
axes[1].set_xlabel('Disorder Type')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=20)
axes[1].legend(title='Gender')

# BMI vs Disorder
bmi_disorder = df.groupby(['Sleep_Disorder_Type', 'BMI_Category']).size().unstack()
bmi_disorder.plot(kind='bar', ax=axes[2], colormap='Set2', edgecolor='white')
axes[2].set_title('BMI Category vs Disorder Type')
axes[2].set_xlabel('Disorder Type')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=20)
axes[2].legend(title='BMI', fontsize=8)

plt.tight_layout()
plt.savefig('eda_02_demographics.png', bbox_inches='tight')
plt.show()
print("Plot 2 saved: eda_02_demographics.png")


# ============================================================
# PLOT 3: SLEEP METRICS BY DISORDER
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('PLOT 3 — Sleep Metrics by Disorder Type', fontsize=14, fontweight='bold')

metrics = ['Sleep_Duration_hrs', 'Quality_of_Sleep_1_10', 'Stress_Level_1_10']
titles  = ['Sleep Duration (hrs)', 'Quality of Sleep (1-10)', 'Stress Level (1-10)']
colors  = list(DISORDER_COLORS.values())

for ax, metric, title in zip(axes, metrics, titles):
    data = [df[df['Sleep_Disorder_Type'] == d][metric].dropna()
            for d in DISORDER_COLORS.keys()]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(list(DISORDER_COLORS.keys()), rotation=20, fontsize=8)
    ax.set_ylabel(title)

plt.tight_layout()
plt.savefig('eda_03_sleep_metrics.png', bbox_inches='tight')
plt.show()
print("Plot 3 saved: eda_03_sleep_metrics.png")


# ============================================================
# PLOT 4: CLINICAL MARKERS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PLOT 4 — Clinical Markers by Disorder', fontsize=14, fontweight='bold')

# AHI Score
for disorder, color in DISORDER_COLORS.items():
    subset = df[df['Sleep_Disorder_Type'] == disorder]['AHI_Score']
    axes[0].hist(subset, bins=20, alpha=0.5, label=disorder, color=color)
axes[0].set_title('AHI Score Distribution')
axes[0].set_xlabel('AHI Score (higher = more severe apnea)')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=8)
axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Mild Apnea (5)')
axes[0].axvline(x=15, color='darkred', linestyle='--', alpha=0.7, label='Moderate (15)')

# SaO2 Level
for disorder, color in DISORDER_COLORS.items():
    subset = df[df['Sleep_Disorder_Type'] == disorder]['SaO2_Level_pct']
    axes[1].hist(subset, bins=20, alpha=0.5, label=disorder, color=color)
axes[1].set_title('SaO2 (Oxygen Saturation) Distribution')
axes[1].set_xlabel('SaO2 Level % (lower = worse)')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=8)
axes[1].axvline(x=92, color='red', linestyle='--', alpha=0.8, label='Critical (92%)')

plt.tight_layout()
plt.savefig('eda_04_clinical_markers.png', bbox_inches='tight')
plt.show()
print("Plot 4 saved: eda_04_clinical_markers.png")


# ============================================================
# PLOT 5: WEARABLE SENSOR DATA
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PLOT 5 — Wearable Sensor Data by Disorder', fontsize=14, fontweight='bold')
axes = axes.flatten()

wearable_cols = ['Wearable_Movement_Actigraphy', 'Wearable_SpO2_pct',
                 'HRV_ms', 'Respiratory_Rate_bpm']
wearable_titles = ['Movement (Actigraphy)', 'Wearable SpO2 %',
                   'HRV (ms)', 'Respiratory Rate (bpm)']

for ax, col, title in zip(axes, wearable_cols, wearable_titles):
    data  = [df[df['Sleep_Disorder_Type'] == d][col].dropna()
             for d in DISORDER_COLORS.keys()]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(list(DISORDER_COLORS.keys()), rotation=20, fontsize=8)
    ax.set_ylabel(title)

plt.tight_layout()
plt.savefig('eda_05_wearable_sensors.png', bbox_inches='tight')
plt.show()
print("Plot 5 saved: eda_05_wearable_sensors.png")


# ============================================================
# PLOT 6: CORRELATION HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(16, 12))
fig.suptitle('PLOT 6 — Feature Correlation Heatmap', fontsize=14, fontweight='bold')

num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5,
            annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Between All Numeric Features', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('eda_06_correlation_heatmap.png', bbox_inches='tight')
plt.show()
print("Plot 6 saved: eda_06_correlation_heatmap.png")


# ============================================================
# PLOT 7: TOP FEATURE IMPORTANCE (using variance)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('PLOT 7 — Feature Variance (Spread/Importance Proxy)',
             fontsize=14, fontweight='bold')

num_df_norm = (num_df - num_df.min()) / (num_df.max() - num_df.min())
variances = num_df_norm.var().sort_values(ascending=True)

colors_bar = ['#e74c3c' if v > variances.median() else '#3498db'
              for v in variances]
bars = ax.barh(variances.index, variances.values,
               color=colors_bar, edgecolor='white')
ax.set_title('Normalized Feature Variance\n(Red = High variance = more informative)',
             fontsize=11)
ax.set_xlabel('Variance (normalized)')
ax.axvline(x=variances.median(), color='black', linestyle='--',
           alpha=0.5, label='Median')
ax.legend()

plt.tight_layout()
plt.savefig('eda_07_feature_variance.png', bbox_inches='tight')
plt.show()
print("Plot 7 saved: eda_07_feature_variance.png")


# ============================================================
# PLOT 8: SCATTER — AHI vs SaO2 (key clinical relationship)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PLOT 8 — Key Clinical Relationships', fontsize=14, fontweight='bold')

# AHI vs SaO2
for disorder, color in DISORDER_COLORS.items():
    subset = df[df['Sleep_Disorder_Type'] == disorder]
    axes[0].scatter(subset['AHI_Score'], subset['SaO2_Level_pct'],
                    alpha=0.4, label=disorder, color=color, s=20)
axes[0].set_title('AHI Score vs SaO2 Level')
axes[0].set_xlabel('AHI Score')
axes[0].set_ylabel('SaO2 Level %')
axes[0].legend(fontsize=8)
axes[0].axhline(y=92, color='red', linestyle='--', alpha=0.5, label='Critical SpO2')

# Sleep Duration vs Quality
for disorder, color in DISORDER_COLORS.items():
    subset = df[df['Sleep_Disorder_Type'] == disorder]
    axes[1].scatter(subset['Sleep_Duration_hrs'], subset['Quality_of_Sleep_1_10'],
                    alpha=0.4, label=disorder, color=color, s=20)
axes[1].set_title('Sleep Duration vs Quality of Sleep')
axes[1].set_xlabel('Sleep Duration (hrs)')
axes[1].set_ylabel('Quality of Sleep (1-10)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('eda_08_scatter_relationships.png', bbox_inches='tight')
plt.show()
print("Plot 8 saved: eda_08_scatter_relationships.png")


# ============================================================
# PLOT 9: OCCUPATION vs DISORDER
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('PLOT 9 — Occupation vs Sleep Disorder', fontsize=14, fontweight='bold')

occ_disorder = df.groupby(['Occupation', 'Sleep_Disorder_Type']).size().unstack(fill_value=0)
occ_disorder.plot(kind='bar', ax=ax,
                  color=[DISORDER_COLORS[c] for c in occ_disorder.columns],
                  edgecolor='white')
ax.set_title('Sleep Disorder Count by Occupation')
ax.set_xlabel('Occupation')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)
ax.legend(title='Disorder', fontsize=9)

plt.tight_layout()
plt.savefig('eda_09_occupation_disorder.png', bbox_inches='tight')
plt.show()
print("Plot 9 saved: eda_09_occupation_disorder.png")


# ============================================================
# PLOT 10: SUMMARY STATISTICS TABLE
# ============================================================

print("\n" + "=" * 60)
print("PLOT 10 — KEY STATISTICAL SUMMARY")
print("=" * 60)

key_cols = ['Sleep_Duration_hrs', 'Quality_of_Sleep_1_10',
            'Stress_Level_1_10', 'AHI_Score',
            'SaO2_Level_pct', 'HRV_ms', 'Heart_Rate_bpm']

summary = df.groupby('Sleep_Disorder_Type')[key_cols].mean().round(2)
print("\nMean values per disorder type:")
print(summary.to_string())

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
print("9 plots saved in your project folder:")
print("  eda_01_class_distribution.png")
print("  eda_02_demographics.png")
print("  eda_03_sleep_metrics.png")
print("  eda_04_clinical_markers.png")
print("  eda_05_wearable_sensors.png")
print("  eda_06_correlation_heatmap.png")
print("  eda_07_feature_variance.png")
print("  eda_08_scatter_relationships.png")
print("  eda_09_occupation_disorder.png")
print("=" * 60)
print("Next Step: Model Building!")
print("=" * 60)
