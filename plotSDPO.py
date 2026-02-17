import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ema(x, alpha=0.9):
    """
    x: 1D array-like
    alpha: smoothing factor in [0,1), higher = smoother
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    return y

def mixed():
    path1 = 'logs/TrainingSDPO/version_7'
    path2 = 'logs/TrainingSDPO/version_8'
    df1 = pd.read_csv(path1+'/metrics.csv')
    df2 = pd.read_csv(path2+'/metrics.csv')
    df2['step'] = df1['step'].max() + df2['step']
    df = pd.concat([df1, df2])
    print(df['step'].max())
    dest_path = './'

    loss = df[['epoch', 'Training_loss']].dropna().groupby('epoch').mean()
    fig, ax = plt.subplots()
    plt.plot(df['epoch'].unique(), loss, label='Training Loss')
    plt.legend()
    ax.tick_params('x', rotation=90)
    plt.title('Losses per epoch')
    plt.savefig(dest_path+'Loss_epoch.png')
    
    plt.show()

    df['step'] = df['step'] // 100
    df_filtered = df[['step', 'Training_loss']].dropna().groupby('step').mean()
    plt.plot(df_filtered, label='Training Loss')
    plt.plot(ema(df_filtered), label='Smoothed Training Loss')
    plt.legend()
    plt.title('Losses per step')
    plt.savefig(dest_path+'Loss_step.png')
    plt.show()

    step0 = df[['step', "Rewards_1_start"]].dropna().groupby('step').mean()["Rewards_1_start"] - df[['step',"Rewards_2_start"]].dropna().groupby('step').mean()["Rewards_2_start"]
    plt.plot(step0, label='step0')
    anchor = df[['step', "Rewards_1_anchor"]].dropna().groupby('step').mean()["Rewards_1_anchor"] - df[['step',"Rewards_2_anchor"]].dropna().groupby('step').mean()["Rewards_2_anchor"]
    plt.plot(anchor, label='anchor')
    last = df[['step',"Rewards_1_last"]].dropna().groupby('step').mean()["Rewards_1_last"] - df[['step',"Rewards_2_last"]].dropna().groupby('step').mean()["Rewards_2_last"]
    plt.plot(last, label='last step')
    plt.legend()
    plt.title('Reward gaps')
    plt.show()

    plt.plot(ema(step0), label='step0')
    plt.plot(ema(anchor), label='anchor')  
    plt.plot(ema(last), label='last step')
    plt.title('Reward gaps smooth')
    plt.legend()
    plt.savefig(dest_path+'Reward gaps smooth.png')
    plt.show()

    corr = df[['step',"corr"]].dropna().groupby('step').mean()
    plt.plot(corr, label='raw')
    plt.plot(ema(corr), label='smooth')
    plt.legend()
    plt.title('Advantage - log_diff correlation')
    plt.savefig(dest_path+'Advantage - log_diff correlation.png')
    plt.show()

    corr_cols = ['corr_late_refine', 'corr_mid_structure', 'corr_early_noise']
    df['step'] //= 10
    df_filtered = df[['step'] + corr_cols].dropna().groupby('step').mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_filtered.T, annot=True, cmap='RdYlGn', center=0)
    plt.title("Correlation Heatmap: Training Progress vs. Diffusion Phase")
    plt.xlabel("Step")
    plt.ylabel("Diffusion Phase")
    plt.savefig(dest_path+"Correlation Heatmap: Training Progress vs. Diffusion Phase.png")
    plt.show()

def normal():
    path = 'logs/TrainingSDPO/version_32'
    df = pd.read_csv(path+'/metrics.csv')

    dest_path = 'logs/TrainingSDPO/version_'+path[path.index('_') +1:]+'/'

    plt.plot(df['Training_loss'], label='Training Loss')
    plt.plot(ema(df['Training_loss']), label='Smoothed Training Loss')
    plt.legend()
    plt.title('Losses per step')
    plt.savefig(dest_path+'Loss_step.png')
    plt.show()

    step0 = df[['step', "Rewards_1_start"]].dropna().groupby('step').mean()["Rewards_1_start"] - df[['step',"Rewards_2_start"]].dropna().groupby('step').mean()["Rewards_2_start"]
    plt.plot(step0, label='step0')
    anchor = df[['step', "Rewards_1_anchor"]].dropna().groupby('step').mean()["Rewards_1_anchor"] - df[['step',"Rewards_2_anchor"]].dropna().groupby('step').mean()["Rewards_2_anchor"]
    plt.plot(anchor, label='anchor')
    last = df[['step',"Rewards_1_last"]].dropna().groupby('step').mean()["Rewards_1_last"] - df[['step',"Rewards_2_last"]].dropna().groupby('step').mean()["Rewards_2_last"]
    plt.plot(last, label='last step')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.title('Reward gaps')
    plt.show()

    plt.plot(ema(step0), label='step0')
    plt.plot(ema(anchor), label='anchor')  
    plt.plot(ema(last), label='last step')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Reward gaps smooth')
    plt.legend()
    plt.savefig(dest_path+'Reward gaps smooth.png')
    plt.show()

    plt.plot(df['log_diff'], label='log_diff')
    plt.plot(df['advantage_diff'], label='Adv_diff')
    plt.title('Log_diff - Adv_diff')
    plt.legend()
    plt.savefig(dest_path+'Log_diff - Adv_diff.png')
    plt.show()

    plt.scatter(df['advantage_diff'], df['log_diff'])
    plt.title('Log_diff - Adv_diff')
    plt.axhline()
    plt.axvline()
    plt.savefig(dest_path+'Scatter.png')
    plt.show()

if __name__ == '__main__':
    normal()