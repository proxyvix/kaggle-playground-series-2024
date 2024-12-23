import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "Premium Amount"
train_df = pd.read_csv("train.csv")
train_df.drop(columns=["id"], inplace=True)

print(train_df.info())
print("=" * 50)
print(train_df.describe())
print("=" * 50)
print(train_df.isna().sum())
print("=" * 50)

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
num_cols = train_df.select_dtypes(exclude="object").columns.tolist()

for col in cat_cols:
    print(train_df[col].value_counts())
    print("=" * 50)

for col in num_cols:
    print(train_df[col].value_counts())
    print("=" * 50)

for col in train_df.columns:
    print(f"Number of unique values in {col}")
    print(train_df[col].nunique())
    print("=" * 50)

fig, axes = plt.subplots(
    nrows=len(num_cols),
    ncols=2,
    figsize=(30, 10 * len(num_cols))
)

palette = sns.color_palette("tab10", len(num_cols))
colors = dict(zip(num_cols, palette))

df_binned = train_df.copy()

# Histogram of numerical data
for i, col in enumerate(num_cols):
    if train_df[col].nunique() > 50:
        discrete = False
    else:
        discrete = True

    sns.histplot(
        data=train_df,
        x=col,
        color=colors[col],
        kde=True,
        ax=axes[i, 0],
        discrete=discrete
    )
    axes[i, 0].set_title(f"Distirbution of {col}")
    sns.despine(ax=axes[i, 0])

    if train_df[col].nunique() <= 10:
        sns.violinplot(
            data=train_df,
            x=col,
            y=target,
            color=colors[col],
            ax=axes[i, 1],
        )
        axes[i, 1].set_title(f"Violin plot of {col}")
        sns.despine(ax=axes[i, 1])
    else:
        df_binned[f"{col}_binned"] = pd.cut(train_df[col], bins=10)
        sns.violinplot(
            data=df_binned,
            x=f"{col}_binned",
            y=target,
            color=colors[col],
            ax=axes[i, 1],
        )
        axes[i, 1].set_title(f"Violin plot of {col}")
        sns.despine(ax=axes[i, 1])

plt.tight_layout()
plt.savefig("num_values.jpg", dpi=300)
plt.show()

filtered_columns = [col for col in cat_cols if col != 'Policy Start Date']

fig, axes = plt.subplots(
    nrows=len(filtered_columns),
    ncols=2,
    figsize=(30, 10 * len(filtered_columns))
)

for i, col in enumerate(filtered_columns):
    sns.countplot(
        data=train_df,
        x=col,
        hue=col,
        palette="tab10",
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"Count of {col}")
    sns.despine(ax=axes[i, 0])

    sns.boxplot(
        data=train_df,
        x=col,
        y=target,
        palette="tab10",
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"Box plot of {col}")
    sns.despine(ax=axes[i, 1])

plt.tight_layout()
plt.savefig("cat_values.jpg", dpi=300)
plt.show()

plt.figure(figsize=(30, 15))

corr = train_df[num_cols].corr()
sns.heatmap(
    data=corr,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.7,
    annot=True
)

plt.savefig("corr_map.jpg", dpi=300)
plt.show()

