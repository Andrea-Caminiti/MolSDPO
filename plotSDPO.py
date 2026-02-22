import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rl_metrics(df, outpath, composed, switch):
    # Load data - handles empty cells as NaN
    
    
    # Updated groups to include your new validation metrics
    groups = {
        'rewards_set_1': ['Rewards_1_start', 'Rewards_1_anchor', 'Rewards_1_last'],
        'rewards_set_2': ['Rewards_2_start', 'Rewards_2_anchor', 'Rewards_2_last'],
        'reward_clena_mean': ['Reward0_mean'],
        'reward_stats': ['Reward_gap_start', 'Reward_gap_last'],
        'training_loss_stats': ['Training_loss'],
        'diffs': ['log_diff', 'advantage_diff'],
        'mol_quality_metrics': ['val/qed', 'val/sa_score', 'val/mol_weight'],
        'mol_diversity_metrics': ['val/novelty', 'val/diversity', 'val/uniqueness', 'val/validity'],
        'stopping_criteria': ['val/stopping_score']
    }

    # 1. Standard Plots for grouped metrics
    for group_name, cols in groups.items():
        if group_name == 'diffs':
            plt.figure(figsize=(10, 5))
            plot_exists = False
            
            for col in cols:
                if col in df.columns:
                    # Drop NaN values for this specific column to avoid breaks in the line
                    valid_df = df[df['step'] > 200].dropna(subset=[col, 'step'])
                    if not valid_df.empty:
                        plt.plot(valid_df['step'], valid_df[col], label=col, alpha=0.8)
                        
                        plot_exists = True
            
            if plot_exists:
                if composed:
                            plt.axvline(switch, color='darkorchid', linestyle='--', ymin=0.0, ymax=1.0, label='Break')
                            plt.legend()
                plt.title(group_name.replace('_', ' ').title())
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                
                # Save using the group name
                save_name = f"{group_name}.png"
                plt.savefig(os.path.join(outpath, save_name))
                print(f"Generated '{save_name}'")
            plt.close()
        else:
            plt.figure(figsize=(10, 5))
            plot_exists = False
            
            for col in cols:
                if col in df.columns:
                    # Drop NaN values for this specific column to avoid breaks in the line
                    valid_df = df[df['step'] > 10].dropna(subset=[col, 'step'])
                    if not valid_df.empty:
                        plt.plot(valid_df['step'], valid_df[col], label=col, alpha=0.8)
                        plot_exists = True
            
            if plot_exists:
                if composed:
                            plt.axvline(switch, color='darkorchid', linestyle='--', ymin=0.0, ymax=1.0, label='Break')
                            plt.legend()
                plt.title(group_name.replace('_', ' ').title())
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                
                # Save using the group name
                save_name = f"{group_name}.png"
                plt.savefig(os.path.join(outpath, save_name))
                print(f"Generated '{save_name}'")
            plt.close()

    # 2. SPECIAL PLOT: Correlation with Smoothed Trend Line
    if 'log_adv_corr' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Drop missing values to ensure the rolling calculation is continuous
        corr_df = df.dropna(subset=['log_adv_corr', 'step']).copy()
        
        if not corr_df.empty:
            # Plot RAW data in the background (faint gray)
            if composed:
                            plt.axvline(switch, color='darkorchid', linestyle='--', ymin=0.0, ymax=1.0, label='Break')
                            plt.legend()
            plt.plot(corr_df['step'], corr_df['log_adv_corr'], 
                     color='gray', alpha=0.3, label='Raw Correlation',
                     marker='o', markersize=3, linestyle='-')
            
            # Calculate Smoothed Line (Rolling Mean)
            # window=15 is good for noisy RL data; increase it for a flatter trend
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
            plt.savefig(os.path.join(outpath, 'correlation_smoothed.png'))
            print("Generated 'correlation_smoothed.png' with trend line.")
        plt.close()

if __name__ == "__main__":
    # Update this path to your actual metrics file
    file_name = 'logs/TrainingSDPO/version_0/metrics.csv' 

    optional_fname = ''
    switch = None
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        composed = False
        if optional_fname:
            df2 = pd.read_csv(optional_fname)
            df['step'] += df2['step'].max() + 1
            df = pd.concat([df2, df])
            composed = True
            switch = df2['step'].max() + 1
        # Set the output directory to where the CSV is located
        outpath = os.path.dirname(file_name)
        plot_rl_metrics(df, outpath, composed, switch)
    else:
        print(f"Error: File '{file_name}' not found. Please check the path.")