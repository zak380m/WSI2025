import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re


# Load and combine all CSV files matching the pattern
csv_files = glob.glob('astar_results_*.csv')  # You can adjust the path or pattern
dfs = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dfs, ignore_index=True)

def to_seconds(x):
    if 'µs' in x:
        return float(x.replace('µs', '')) / 1_000_000
    elif 'ms' in x:
        return float(x.replace('ms', '')) / 1000
    elif 'm' in x and 's' in x:
        match = re.match(r'(?P<min>\d+)m(?P<sec>[\d\.]+)', x)
        if match:
            return float(match.group('min')) * 60 + float(match.group('sec'))
        else:
            raise ValueError(f"Unrecognized time format: {x}")
    elif 's' in x:
        return float(x.replace('s', ''))
    else:
        raise ValueError(f"Unrecognized time format: {x}")

df['time_taken_sec'] = df['time_taken'].apply(to_seconds)


# Group by mode and heuristic for statistics
grouped_stats = df.groupby(['mode', 'heuristic']).agg({
    'time_taken_sec': ['mean', 'min', 'max'],
    'visited_states': ['mean', 'min', 'max']
}).reset_index()

# Flatten multi-index columns
grouped_stats.columns = ['_'.join(col).strip('_') for col in grouped_stats.columns.values]

# Map mode numbers to labels
mode_labels = {3: '3x3', 4: '4x4 (k>20)', 5: 'All 4x4'}
grouped_stats['mode_label'] = grouped_stats['mode'].map(mode_labels)

# Get unique heuristics and modes
heuristics = df['heuristic'].unique()
modes = grouped_stats['mode_label'].unique()

# Plot 1: Time taken ranges with averages with proper y-axis scaling
plt.figure(figsize=(12, 6))
x = np.arange(len(modes))
width = 0.2

# Find the overall min and max values for scaling
time_min = grouped_stats['time_taken_sec_min'].min()
time_max = grouped_stats['time_taken_sec_max'].max()

for i, heuristic in enumerate(heuristics):
    subset = grouped_stats[grouped_stats['heuristic'] == heuristic]
    
    # Plot range (min to max)
    bars = plt.bar(x + i*width, 
                  subset['time_taken_sec_max'] - subset['time_taken_sec_min'], 
                  width, 
                  bottom=subset['time_taken_sec_min'],
                  alpha=0.5,
                  label=heuristic)
    
    # Plot average line with value label
    for j, (mode, mean_val) in enumerate(zip(modes, subset['time_taken_sec_mean'])):
        # Draw the average line
        plt.hlines(mean_val, 
                  x[j] + i*width - width/2, 
                  x[j] + i*width + width/2, 
                  colors='black', 
                  linewidth=2)
        
        # Format label based on value size
        if mean_val < 0.001:  # microseconds range
            label_text = f'{mean_val*1e6:.0f}µs'
        elif mean_val < 1:     # milliseconds range
            label_text = f'{mean_val*1e3:.0f}ms'
        else:                  # seconds range
            label_text = f'{mean_val:.2f}s'
        
        # Position label above the line
        plt.text(x[j] + i*width + width/2 + 0.02, 
                mean_val * 1.2,  # Position slightly above
                label_text,
                va='bottom', ha='left', fontsize=9)

plt.title('Time Taken Ranges by Heuristic and Mode')
plt.ylabel('Time (log scale)')
plt.xticks(x + width*(len(heuristics)-1)/2, modes)

# Set explicit log scale with 10^(-4) included
plt.yscale('log')
plt.ylim(max(1e-6, time_min * 0.5), time_max * 2)  # Add some padding
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])  # Explicit ticks including 10^(-4)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f'{x*1e6:.0f}µs' if x < 1e-3 else 
                (f'{x*1e3:.0f}ms' if x < 1 else f'{x:.2f}s')))

plt.grid(True, alpha=0.3, which='both')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot 2: Visited states ranges with averages
plt.figure(figsize=(12, 6))
x = np.arange(len(modes))
width = 0.2  # Wider bars for better visibility

for i, heuristic in enumerate(heuristics):
    subset = grouped_stats[grouped_stats['heuristic'] == heuristic]
    
    # Plot range (min to max)
    plt.bar(x + i*width, 
            subset['visited_states_max'] - subset['visited_states_min'], 
            width, 
            bottom=subset['visited_states_min'],
            alpha=0.5,
            label=heuristic)
    
    # Plot average line with value label
    for j, (mode, mean_val) in enumerate(zip(modes, subset['visited_states_mean'])):
        # Draw the average line
        plt.hlines(mean_val, 
                   x[j] + i*width - width/2, 
                   x[j] + i*width + width/2, 
                   colors='black', 
                   linewidth=2)
        
        # Add text label
        plt.text(x[j] + i*width + width/2 + 0.02, 
                 mean_val, 
                 f'{mean_val:,.0f}',
                 va='center', ha='left', fontsize=9)

plt.title('Visited States Ranges by Heuristic and Mode')
plt.ylabel('Number of States')
plt.xlabel('Puzzle Mode')
plt.xticks(x + width*(len(heuristics)-1)/2, modes)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# Keep the solution length distribution plots as they were
for mode in [3, 4, 5]:
    plt.figure(figsize=(10, 6))
    mode_data = df[(df['mode'] == mode) & (df['heuristic'] == 'Manhattan distance')]
    
    # Calculate statistics
    avg_length = mode_data['solution_length'].mean()
    min_length = int(mode_data['solution_length'].min())
    max_length = int(mode_data['solution_length'].max())
    
    # Create bins of size 2 units, extending slightly beyond the range
    bin_size = 2
    bins = range(min_length, max_length + bin_size + 1, bin_size)
    
    # Plot histogram
    plt.hist(mode_data['solution_length'], bins=bins, alpha=0.6, color='skyblue')
    
    # Add average line and labels
    plt.axvline(avg_length, color='red', linestyle='--', 
                label=f'Average: {avg_length:.1f} steps')
    plt.title(f'Solution Length Distribution - {mode_labels.get(mode, f"{mode}x{mode} Puzzle")}')
    plt.xlabel('Number of Steps')
    plt.ylabel('Number of Trials')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()