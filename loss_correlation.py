import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_tb_log(path, tag='loss'):
    ea = EventAccumulator(path)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        raise ValueError(f"Tag '{tag}' not found")
    
    df = pd.DataFrame(ea.Scalars(tag))
    df = df[['step', 'value']]
    
    # FIX: Group by step and take the mean to handle 
    # duplicate entries/multiple event files
    df = df.groupby('step').mean().reset_index()
    
    return df.rename(columns={'value': path})

def plot_correlation(run1_path, run2_path, tag='loss'):
    # Extract data
    df1 = get_tb_log(run1_path, tag)
    df2 = get_tb_log(run2_path, tag)
    
    # Merge on 'step' to ensure we compare the same points in training
    merged_df = pd.merge(df1, df2, on='step', suffixes=('_run1', '_run2'))
    
    # Calculate Correlation Coefficient
    correlation = merged_df.iloc[:, 1].corr(merged_df.iloc[:, 2])
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.regplot(x=merged_df.columns[1], y=merged_df.columns[2], data=merged_df, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title(f'Correlation Plot: {tag}\nPearson Correlation: {correlation:.4f}')
    plt.xlabel(f'Run 1 {tag}')
    plt.ylabel(f'Run 2 {tag}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Usage
# plot_correlation('path/to/run1', 'path/to/run2', tag='train/loss')
plot_correlation("runs/simple/optimized5", "runs/simple/1")