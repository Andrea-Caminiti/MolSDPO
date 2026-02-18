import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics(csv_path):
    # 1. Load the data
    # Handle missing values: pandas automatically converts empty cells to NaN
    df = pd.read_csv(csv_path)
    outpath = csv_path.replace(os.path.basename(csv_path), "")
    # 2. Define metric groups for better organization
    train_losses = [
        'loss', 'coord_loss', 'type_loss', 'dist_loss', 
        'geom_loss', 'com_loss', 'mag_loss'
    ]
    val_denoise = [
        'val/denoise_loss_t10', 'val/denoise_loss_t100', 
        'val/denoise_loss_t500', 'val/denoise_loss_t900'
    ]
    val_ratios = [
        'val/valid_ratio', 'val/connected_ratio', 'val/realistic_ratio'
    ]
    val_stats = ['val/mean_atoms', 'val/mean_min_dist_A']

    # Helper function to plot a group of metrics
    def plot_group(cols, title, ylabel, filename, marker=None):
        plt.figure(figsize=(12, 6))
        for col in cols:
            if col in df.columns:
                # Drop missing values specifically for this column to ensure a continuous line
                valid_data = df.dropna(subset=[col, 'step'])
                plt.plot(valid_data['step'], valid_data[col], label=col, marker=marker)
        
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()

    # 3. Generate the plots
    # Plot Training Losses
    plot_group(train_losses, 'Training Losses', 'Loss Value', f'{outpath}/training_losses.png')

    # Plot Validation Denoise Losses (with markers to see individual eval points)
    plot_group(val_denoise, 'Validation Denoise Losses', 'Loss Value', f'{outpath}/val_denoise_losses.png', marker='o')

    # Plot Validation Quality Ratios
    plot_group(val_ratios, 'Validation Quality Ratios', 'Ratio (0-1)', f'{outpath}/val_ratios.png', marker='s')

    # Plot Mixed Validation Stats (Double Y-Axis)
    if all(col in df.columns for col in val_stats):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot Mean Atoms on Left Axis
        d1 = df.dropna(subset=['val/mean_atoms', 'step'])
        ax1.plot(d1['step'], d1['val/mean_atoms'], color='tab:blue', marker='^', label='Mean Atoms')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Mean Atoms', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Plot Min Dist on Right Axis
        ax2 = ax1.twinx()
        d2 = df.dropna(subset=['val/mean_min_dist_A', 'step'])
        ax2.plot(d2['step'], d2['val/mean_min_dist_A'], color='tab:red', marker='d', label='Mean Min Dist (A)')
        ax2.set_ylabel('Mean Min Dist (A)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Mean Atoms and Min Distance')
        fig.tight_layout()
        plt.savefig(f'{outpath}/val_stats.png')
        print("Saved: val_stats.png")
        plt.close()

if __name__ == "__main__":
    # Path to your CSV file
    csv_file = 'logs/Pretrain/version_6/metrics.csv'
    
    if os.path.exists(csv_file):
        plot_training_metrics(csv_file)
    else:
        print(f"Error: {csv_file} not found.")