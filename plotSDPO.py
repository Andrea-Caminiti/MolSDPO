import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rl_metrics(csv_path):
    # Load data - handles empty cells as NaN
    df = pd.read_csv(csv_path)
    
    
    outpath = csv_path.replace(os.path.basename(csv_path), "")
    # Define groups for general plotting
    groups = {
        'rewards_1': ['Rewards_1_start', 'Rewards_1_anchor', 'Rewards_1_last'],
        'rewards_2': ['Rewards_2_start', 'Rewards_2_anchor', 'Rewards_2_last'],
        'reward_stats': ['Reward0_mean', 'Reward_gap_start', 'Reward_gap_last'],
        'training_stats': ['Training_loss', 'log_diff', 'advantage_diff']
    }

    # 1. Standard Plots for other metrics
    for name, cols in groups.items():
        plt.figure(figsize=(10, 5))
        for col in cols:
            if col in df.columns:
                valid_df = df.dropna(subset=[col, 'step'])
                plt.plot(valid_df['step'], valid_df[col], label=col, alpha=0.8)
        plt.title(name.replace('_', ' ').title())
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'{outpath}/{name}.png')
        plt.close()

    # 2. SPECIAL PLOT: Correlation with Smoothed Trend Line
    if 'log_adv_corr' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Drop missing values to ensure the rolling calculation is continuous
        corr_df = df.dropna(subset=['log_adv_corr', 'step']).copy()
        
        # Plot RAW data in the background (faint gray with markers)
        plt.plot(corr_df['step'], corr_df['log_adv_corr'], 
                 color='gray', alpha=0.3, label='Raw Correlation',
                 marker='o', markersize=3, linestyle='-')
        
        # Calculate Smoothed Line (Rolling Mean)
        # Change window=15 to a larger number (e.g., 50) for more aggressive smoothing
        corr_df['smoothed'] = corr_df['log_adv_corr'].rolling(window=15, min_periods=1, center=True).mean()
        
        # Plot SMOOTHED line in the foreground (bold red)
        plt.plot(corr_df['step'], corr_df['smoothed'], 
                 color='red', linewidth=2.5, label='Smoothed Trend (Rolling Mean)')
        
        plt.title('Log-Advantage Correlation: Raw vs. Trend', fontsize=14)
        plt.xlabel('Step')
        plt.ylabel('Correlation Coefficient')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'{outpath}/correlation_smoothed.png')
        print("Generated 'correlation_smoothed.png' with trend line.")
        plt.close()

if __name__ == "__main__":
    file_name = 'logs/TrainingSDPO/version_0/metrics.csv' # Ensure this matches your filename
    if os.path.exists(file_name):
        plot_rl_metrics(file_name)
    else:
        print(f"File '{file_name}' not found.")

