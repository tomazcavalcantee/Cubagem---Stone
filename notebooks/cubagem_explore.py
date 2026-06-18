import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %% 
# Parameters for the underlying normal distribution
mu = 1.0       # Centered at 1
sigma = 0.5    # Standard deviation (spread)

# Generate points for the x-axis (must be > 0 for lognormal values)
y = np.linspace(0.01, 10, 1000)

# Theoretical Lognormal PDF formula
pdf = (1 / (y * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(y) - mu) ** 2) / (2 * sigma ** 2))

# Calculate Mode, Median, and Mean
mode = np.exp(mu - sigma**2)
median = np.exp(mu)
mean = np.exp(mu + (sigma**2) / 2)

# Plotting the distribution curve
plt.plot(y, pdf, label=f'Lognormal PDF ($\mu=1, \sigma={sigma}$)', color='darkblue', linewidth=2.5)

# Add vertical lines to show the separation of Mode, Median, and Mean
plt.axvline(x=mode, color='crimson', linestyle='--', linewidth=1.5, label=f'Mode ({mode:.2f})')
plt.axvline(x=median, color='orange', linestyle='-.', linewidth=1.5, label=f'Median ({median:.2f})')
plt.axvline(x=mean, color='forestgreen', linestyle=':', linewidth=2, label=f'Mean ({mean:.2f})')

# Labels and Styling
plt.title('The Exponential of a Normal Distribution ($\mu = 1$)')
plt.xlabel('Value ($y = e^X$)')
plt.ylabel('Probability Density')
plt.xlim(0, 8)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

# Save the plot
plt.savefig('lognormal_visualization.png', bbox_inches='tight')
plt.close()




# %%
# Load your dataset
df = pd.read_csv("cubagem_40k_amazon.csv")
# df = pd.read_csv("cubagem_40k_amazon_train.csv")


total_products = len(df)
print(f"Total number of products (rows): {total_products}")
rows, columns = df.shape
print(f"The dataset has {rows} rows and {columns} columns.")

# Check for missing values and unique counts in category columns
print("Missing Values per Column:")
print(df[['main_category', 'source_category', 'categories', 'n_categories_levels']].isnull().sum())

print("\nUnique Counts:")
print(df[['main_category', 'source_category', 'n_categories_levels']].nunique())
# %%

# ==========================================
# 0. CONFIGURE GLOBAL TEXT SIZES (ALL 25)
# ==========================================
plt.rcParams['axes.titlesize'] = 25     # Subplot titles
plt.rcParams['axes.labelsize'] = 25     # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 20    # X-axis category names
plt.rcParams['ytick.labelsize'] = 25    # Y-axis numbers/ticks


# 2. Define the category column name
category_col = 'main_category' 

if category_col not in df.columns:
    print(f"Error: Column '{category_col}' not found. Dataset columns:")
    print(df.columns.tolist())
    exit()

# 3. Calculate log-transformed columns
df['log_length'] = np.log(df['length_cm'])
df['log_height'] = np.log(df['height_cm'])
df['log_width'] = np.log(df['width_cm'])
df['log_weight'] = np.log(df['weight_g'])

# 4. Filter for the top 5 categories
top_5_categories = df[category_col].value_counts().nlargest(5).index
df_filtered = df[df[category_col].isin(top_5_categories)]

# 5. Columns to map across the 1x4 grid
log_cols = ['log_length', 'log_height', 'log_width', 'log_weight']

# 6. Generate the 1 row x 4 columns grid of boxplots
# Height bumped to 10 to prevent large, rotated category text from clipping out
fig, axes = plt.subplots(1, 4, figsize=(28, 10))

for i, col in enumerate(log_cols):
    sns.boxplot(
        data=df_filtered, 
        x=category_col, 
        y=col, 
        ax=axes[i], 
        color='royalblue',
        order=top_5_categories
    )
    # Titles and labels inherit size 25 from rcParams; setting formatting adjustments here
    axes[i].set_title(f'{col} Distribution', fontweight='bold', pad=15)
    # axes[i].set_xlabel('Category', labelpad=15)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Log Scale Value', labelpad=15)
    
    # Rotate category labels 45 degrees so size 25 font doesn't collide
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

# Add extra space around subplots so massive labels do not touch adjacent charts
plt.tight_layout(pad=3.0)

# 7. Save the figure as a vector file (SVG format)
vector_filename = 'categories_large_boxplots.svg'
plt.savefig(vector_filename, format='svg', bbox_inches='tight')
print(f"\nBox plots successfully saved as a vector file: '{vector_filename}'")
# %% volume_weight_distributions

# ==========================================
# 0. CONFIGURE GLOBAL TEXT SIZES
# ==========================================
plt.rcParams['axes.titlesize'] = 25     # Increased and balanced (was 56)
plt.rcParams['axes.labelsize'] = 25     # Font size of X and Y axis labels
plt.rcParams['xtick.labelsize'] = 25    # Font size of X-axis tick markers
plt.rcParams['ytick.labelsize'] = 25    # Font size of Y-axis tick markers

# 2. Filter for your specific columns
target_cols = ['volume_cm3', 'weight_g', 'log_volume', 'log_weight']
available_cols = [col for col in target_cols if col in df.columns]

# 3. Print Max, Min, and Median
stats = df[available_cols].describe().loc[['min', 'max', 'mean', '50%']]
stats.rename(index={'50%': 'median'}, inplace=True)
print("--- SUMMARY STATISTICS ---")
print(stats.T)

# 4. Plot Distributions (2x2 Grid)
print("\nGenerating distribution plots...")
# Slightly bumped figsize to (16, 11) to give the larger text breathing room
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Flatten axes array for easy looping
axes = axes.flatten()

for i, col in enumerate(available_cols):
    sns.histplot(df[col], kde=True, bins=40, color='royalblue', ax=axes[i])
    # REMOVED fontsize=12 so it inherits the global rcParams size (18)
    axes[i].set_title(f'Distribution of {col}', fontweight='bold', pad=12)
    axes[i].set_xlabel('Value', labelpad=8) # Added standard x-label for clarity
    axes[i].set_ylabel('Frequency', labelpad=8)

plt.tight_layout()

# REMEMBER: savefig MUST come before plt.show() if you uncomment it!
plt.savefig('volume_weight_distributions.svg', format='svg', bbox_inches='tight')
plt.show()
# %%
import numpy as np
# 2. Transform dimensions into log values
df['log_length'] = np.log(df['length_cm'])
df['log_width'] = np.log(df['width_cm'])
df['log_height'] = np.log(df['height_cm'])

# 3. Export the transformed data to a new CSV file
transformed_filename = 'cubagem_40k_amazon_train_transformed.csv'
df.to_csv(transformed_filename, index=False)
print(f"Transformed data successfully saved to '{transformed_filename}'\n")

# 3. Re-order columns to map perfectly to a 2x3 grid (Row 1: Raw, Row 2: Log)
grid_cols = [
    'length_cm', 'width_cm', 'height_cm',  # Row 1 (Top)
    'log_length', 'log_width', 'log_height' # Row 2 (Bottom)
]

# 4. Extract and print summary statistics for confirmation
stats = df[grid_cols].describe().loc[['min', 'max', 'mean', '50%']]
stats.rename(index={'50%': 'median'}, inplace=True)
print("--- SUMMARY STATISTICS ---")
print(stats.T)

# 5. Generate and save the distribution plots in a 2x3 grid (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Flattens into a 1D array of 6 elements (0 to 5)

for i, col in enumerate(grid_cols):
    # Maintaining the royalblue color scheme
    sns.histplot(df[col], kde=True, bins=40, color='royalblue', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()

# 7. Save the figure as a vector file (PDF format)
vector_filename = 'dimensions_distributions.svg'
plt.savefig(vector_filename, format='svg', bbox_inches='tight', dpi=300)
print(f"\nDistribution graphics successfully saved as a vector file: '{vector_filename}'")

# %%
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. CONFIGURE GLOBAL TEXT SIZES (ALL 25)
# ==========================================
plt.rcParams['axes.titlesize'] = 25     # Subplot titles
plt.rcParams['axes.labelsize'] = 25     # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 25    # X-axis numbers/ticks
plt.rcParams['ytick.labelsize'] = 25    # Y-axis numbers/ticks

# 1. Load data (Ensuring df exists for the snippet)
try:
    df = pd.read_csv('cubagem_40k_amazon_train.csv')
except FileNotFoundError:
    print("Error: 'cubagem_40k_amazon_train.csv' not found. Please verify the path.")
    exit()

# 2. Transform dimensions into log values
df['log_length'] = np.log(df['length_cm'])
df['log_width'] = np.log(df['width_cm'])
df['log_height'] = np.log(df['height_cm'])

# 3. Export the transformed data to a new CSV file
transformed_filename = 'cubagem_40k_amazon_train_transformed.csv'
df.to_csv(transformed_filename, index=False)
print(f"Transformed data successfully saved to '{transformed_filename}'\n")

# 3. Re-order columns to map perfectly to a 2x3 grid (Row 1: Raw, Row 2: Log)
grid_cols = [
    'length_cm', 'width_cm', 'height_cm',  # Row 1 (Top)
    'log_length', 'log_width', 'log_height' # Row 2 (Bottom)
]

# 4. Extract and print summary statistics for confirmation
stats = df[grid_cols].describe().loc[['min', 'max', 'mean', '50%']]
stats.rename(index={'50%': 'median'}, inplace=True)
print("--- SUMMARY STATISTICS ---")
print(stats.T)

# 5. Generate and save the distribution plots in a 2x3 grid
# Enlarged figsize to (26, 16) so size 25 font has plenty of breathing room
fig, axes = plt.subplots(2, 3, figsize=(26, 16))
axes = axes.flatten()  # Flattens into a 1D array of 6 elements (0 to 5)

for i, col in enumerate(grid_cols):
    # Maintaining the royalblue color scheme
    sns.histplot(df[col], kde=True, bins=40, color='royalblue', ax=axes[i])
    
    # REMOVED local 'fontsize=12' so it correctly inherits global size 25
    axes[i].set_title(f'Distribution of {col}', fontweight='bold', pad=15)
    axes[i].set_xlabel('Value', labelpad=12)
    axes[i].set_ylabel('Frequency', labelpad=12)

# Adjust padding explicitly to prevent the large text from clipping out of bounds
plt.tight_layout(pad=3.0)

# 7. Save the figure as a vector file (SVG format matching your extension choice)
vector_filename = 'dimensions_distributions.svg'
plt.savefig(vector_filename, format='svg', bbox_inches='tight')
print(f"\nDistribution graphics successfully saved as a vector file: '{vector_filename}'")

# %%
import numpy as np
# 1. Force pandas to display all rows so nothing gets truncated with ellipses (...)
pd.set_option('display.max_rows', None)

# 2. Calculate raw counts and percentages of missing values
missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100

# 3. Combine both into a clean, readable summary table
missing_data = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage (%)': missing_percentages
})

# 4. Sort the table in descending order of missing values
missing_data_sorted = missing_data.sort_values(by='Missing Count', ascending=False)

print("--- Missing Values Per Column ---")
print(missing_data_sorted)
# %% main category

# 1. Frequency and Percentage Distribution

# Tell pandas to display all rows without truncation
pd.set_option('display.max_rows', None)

# 1. Print all categories with their raw counts
print("--- All Main Categories (Counts) ---")
print(df['main_category'].value_counts())

# 2. Print all categories with their percentage contribution
print("\n--- All Main Categories (Percentages) ---")
print(df['main_category'].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# Find out exactly how many unique categories exist
num_categories = df['main_category'].nunique()

# Dynamically set height: allocate 0.3 inches per category, minimum of 6 inches
fig_height = max(6, num_categories * 0.3)

# Set the figure size using the dynamic height
plt.figure(figsize=(12, fig_height))

# Plot all categories sorted in descending order of frequency
sns.countplot(
    data=df, 
    y='main_category', 
    order=df['main_category'].value_counts().index,
    palette='viridis'
)

plt.title(f'Distribution of All {num_categories} Main Categories', fontsize=14, pad=15)
plt.xlabel('Number of Products', fontsize=12)
plt.ylabel('Main Category', fontsize=12)

# Adjust layout to make sure text labels aren't cut off
plt.tight_layout()
plt.savefig('main_categories.png', bbox_inches='tight', dpi=300)
plt.show()
# %% Source Category Analysis

# 1. Frequency and Percentage Distribution

# Tell pandas to display all rows without truncation
pd.set_option('display.max_rows', None)

# 1. Print all source categories with their raw counts
print("--- All Source Categories (Counts) ---")
print(df['source_category'].value_counts())

# 2. Print all source categories with their percentage contribution
print("\n--- All Source Categories (Percentages) ---")
print(df['source_category'].value_counts(normalize=True) * 100)


# 2. Visual Distribution Chart

# Find out exactly how many unique source categories exist
num_source_categories = df['source_category'].nunique()

# Dynamically set height: allocate 0.3 inches per category, minimum of 6 inches
fig_height = max(6, num_source_categories * 0.3)

# Set up the plot layout using the dynamic height
fig, ax = plt.subplots(figsize=(12, fig_height))

# Plot all source categories sorted in descending order of frequency
sns.countplot(
    data=df, 
    y='source_category', 
    order=df['source_category'].value_counts().index,
    palette='viridis',
    ax=ax
)

# Customize title, labels, and font sizes
ax.set_title(f'Distribution of All {num_source_categories} Source Categories', fontsize=14, pad=15)
ax.set_xlabel('Number of Products', fontsize=12)
ax.set_ylabel('Source Category', fontsize=12)

# Adjust layout to make sure text labels aren't cut off
plt.tight_layout()
plt.show()

# %% # %% Store Analysis (Top 20)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Print the total number of unique stores
total_stores = df['store'].nunique()
print(f"Total number of unique stores in dataset: {total_stores}\n")

# 2. Isolate the top 20 most frequent stores
top_20_counts = df['store'].value_counts().head(20)
top_20_pct = df['store'].value_counts(normalize=True).head(20) * 100

# 3. Print top 20 stores with their raw counts
print("--- Top 20 Stores (Counts) ---")
print(top_20_counts)

# 4. Print top 20 stores with their percentage contribution
print("\n--- Top 20 Stores (Percentages) ---")
print(top_20_pct)


# 5. Visual Distribution Chart for Top 20 Stores

# Set a fixed, clean figure size appropriate for exactly 20 items
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the top 20 stores sorted in descending order of frequency
sns.countplot(
    data=df, 
    y='store', 
    order=top_20_counts.index,
    palette='viridis',
    ax=ax
)

# Customize title, labels, and font sizes
ax.set_title(f'Distribution of Top 20 Stores (Out of {total_stores} Total)', fontsize=14, pad=15)
ax.set_xlabel('Number of Products', fontsize=12)
ax.set_ylabel('Store', fontsize=12)

# Adjust layout to make sure text labels aren't cut off
plt.tight_layout()

# Save the high-resolution visualization directly to a file
plt.savefig('top_20_stores_distribution.pdf', bbox_inches='tight', dpi=300)
# %% # 2. EXTRACT LEAF CATEGORIES
# ==========================================
# Since entries can have multiple category trees per row, we split them first.
# Standard ecommerce sets use ';' or ',' to separate multiple paths. 
# We use a regex split '[;,]' to automatically catch either a comma or a semicolon.
df['split_paths'] = df['categories'].fillna('').str.split(r'[;,]')

# "Explode" turns lists of paths into individual rows so every path gets evaluated
df_exploded = df.explode('split_paths')

# Clean up any trailing/leading whitespaces from the paths
df_exploded['split_paths'] = df_exploded['split_paths'].str.strip()

# Function to safely extract the last item (leaf) after the '>' delimiter
def extract_leaf(path):
    if not path:
        return None
    parts = path.split('>')
    return parts[-1].strip() if parts else None

# Apply the function to get the leaf category for each path
df_exploded['leaf_category'] = df_exploded['split_paths'].apply(extract_leaf)

# Filter out any empty rows or missing values
df_leaf = df_exploded[df_exploded['leaf_category'].str.len() > 0].copy()

# ==========================================
# 3. AGGREGATE AND SORT
# ==========================================
# Count frequencies of each leaf category (value_counts automatically sorts descending)
leaf_counts = df_leaf['leaf_category'].value_counts().reset_index()
leaf_counts.columns = ['leaf_category', 'count']

print(f"Total unique leaf categories found: {len(leaf_counts)}")
print("\nTop 20 most frequent leaf categories:")
print(leaf_counts.head(20))

# ==========================================
# 4. VISUALIZE THE DISTRIBUTION
# ==========================================
# If you have hundreds of leaves, plotting all of them makes the chart unreadable.
# We will isolate the Top N categories for the visualization.
top_n = 100
plot_data = leaf_counts.head(top_n)

# Dynamically scale the vertical height of the chart based on the number of categories 
# to ensure that labels are readable and do not overlap.
fig_height = max(6, top_n * 0.35)
fig, ax = plt.subplots(figsize=(12, fig_height))

# Create a horizontal bar chart
sns.barplot(
    data=plot_data,
    x='count',
    y='leaf_category',
    ax=ax,
    palette='viridis'
)

# Customize title, labels, and font sizes
ax.set_title(f'Distribution of Top {top_n} Leaf Categories', fontsize=14, pad=15)
ax.set_xlabel('Product Count', fontsize=12)
ax.set_ylabel('Leaf Category', fontsize=12)

# Adjust plot layout to make sure text labels aren't truncated
plt.tight_layout()

# Save the visualization to a high-resolution PNG file
output_filename = 'leaf_categories_distribution.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)

print(f"\nVisualization successfully saved to your environment as '{output_filename}'")

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 2. COUNT THE ">" SIGNS
# ==========================================
# We fill missing (NaN) values with an empty string to avoid string operation errors
df['arrow_count'] = df['categories'].fillna('').str.count('>')

# Print textual breakdown to the console
print("--- Frequency Table of '>' Counts ---")
count_frequency = df['arrow_count'].value_counts().sort_index()
for count, freq in count_frequency.items():
    levels = count + 1 if count > 0 or df[df['arrow_count'] == count]['categories'].dropna().str.len().max() > 0 else 0
    print(self_check := f"{count} '>' signs ({levels} hierarchy levels): {freq} products")

# ==========================================
# 3. PLOT THE DISTRIBUTION
# ==========================================
# Set up the plot layout cleanly without using .figure()
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar chart sorted in ascending order of the count value (0, 1, 2...)
sns.countplot(
    data=df,
    x='arrow_count',
    order=count_frequency.index,
    palette='magma',
    ax=ax
)

# Enhance readability and prevent label overlapping
ax.set_title("Distribution of '>' Sign Counts in Product Categories", fontsize=14, pad=15)
ax.set_xlabel("Number of '>' Signs per Entry", fontsize=12)
ax.set_ylabel("Product Count", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout spacing to ensure no text or labels are truncated
plt.tight_layout()

# Save the visualization directly to a high-resolution file
output_image = 'categories_arrow_count_distribution.png'
plt.savefig(output_image, bbox_inches='tight', dpi=300)

print(f"\nVisualization successfully saved as '{output_image}'")


# %% demo gradient boosting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Generate Synthetic Data (Price vs Weight)
np.random.seed(42)
X_price = np.sort(5 * np.random.rand(80, 1), axis=0)
# A non-linear relationship: Weight increases, dips, then spikes
y_weight = np.sin(X_price).ravel() + np.sin(2 * X_price).ravel() + np.random.normal(0, 0.1, X_price.shape[0])

# X grid for smooth plotting predictions
X_grid = np.linspace(0, 5, 500).reshape(-1, 1)

# 2. Simulate the Boosting Process for 20 Iterations
learning_rate = 0.3
total_steps = 20

# Iteration 0: Start with a baseline prediction (the mean weight of all products)
F_0 = np.mean(y_weight)
F_t_grid = np.full_like(X_grid, F_0)
F_t_train = np.full_like(X_price, F_0)

# Milestone steps requested for visualization
milestones = [1, 2, 10, 20]

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
fig.suptitle("The Evolution of GBDT Over 20 Steps", fontsize=16, fontweight='bold')

plot_row = 0
for t in range(1, total_steps + 1):
    # Step A: Calculate the mistakes/residuals left behind by the current ensemble
    residuals = y_weight - F_t_train.ravel()
    
    # Step B: Train a new tree (weak learner) to predict those residuals
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X_price, residuals)
    
    # Predictions of this specific tree
    tree_pred_grid = tree.predict(X_grid)
    tree_pred_train = tree.predict(X_price)
    
    # Step C: Update the overall ensemble prediction
    F_t_grid = F_t_grid + learning_rate * tree_pred_grid.reshape(-1, 1)
    F_t_train = F_t_train + learning_rate * tree_pred_train.reshape(-1, 1)
    
    # Only plot if the current iteration matches our milestone
    if t in milestones:
        # --- LEFT COLUMN: The Tree's Correction ---
        ax_left = axes[plot_row, 0]
        ax_left.scatter(X_price, residuals, color='red', alpha=0.5, label='Remaining Errors')
        ax_left.plot(X_grid, tree_pred_grid, color='darkred', linestyle='--', linewidth=2, 
                     label=f'Tree {t} Buckets')
        ax_left.set_ylabel(f'Iteration {t}', fontsize=12, fontweight='bold')
        ax_left.grid(True, alpha=0.3)
        if plot_row == 0:
            ax_left.set_title("New Tree Fixing the Current Mistakes", fontsize=13, pad=10)
        ax_left.legend(loc='upper right')
        
        # --- RIGHT COLUMN: The Cumulative Model ---
        ax_right = axes[plot_row, 1]
        ax_right.scatter(X_price, y_weight, color='blue', alpha=0.4, label='Actual Products')
        ax_right.plot(X_grid, F_t_grid, color='green', linewidth=3, label='Combined GBDT Model')
        ax_right.grid(True, alpha=0.3)
        if plot_row == 0:
            ax_right.set_title("The Final Combined Model Curve", fontsize=13, pad=10)
        ax_right.legend(loc='upper right')
        
        plot_row += 1

# Label the bottom axes
axes[3, 0].set_xlabel('Product Price ($)', fontsize=12)
axes[3, 1].set_xlabel('Product Price ($)', fontsize=12)

plt.tight_layout()
plt.show()