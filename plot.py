import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == '__main__':
    path = 'logs/Pretrain/version_0'
    df = pd.read_csv(path+'/metrics.csv')

    dest_path = 'logs/Pretrain/version_'+path[path.index('_') +1:]+'/'
    loss = [df['loss'][df['epoch']==i].mean() for i in range(max(df['epoch']) + 1)]
    fig, ax = plt.subplots()
    epochs = df['epoch'].unique()
    plt.plot(epochs, loss)
    ax.tick_params('x', rotation=90)
    plt.title('Loss_epoch')
    plt.savefig(dest_path+'Loss_epoch.png')
    plt.show()

    plt.title('Loss_step')
    plt.plot(df['step'], df['loss'])
    plt.savefig(dest_path+'Loss_step.png')
    plt.show()
    
    loss_atom = [df['atom_loss'][df['epoch']==i].mean() for i in range(max(df['epoch']) + 1)]
    fig, ax = plt.subplots()
    plt.plot(epochs, loss_atom)
    ax.tick_params('x', rotation=90)

    plt.title('Atom_Loss_epoch')
    plt.savefig(dest_path+'Atom_Loss_epoch.png')
    plt.show()

    plt.title('Atom_Loss_step')
    plt.plot(df['step'], df['atom_loss'])
    plt.savefig(dest_path+'Atom_Loss_step.png')
    plt.show()

    coord_loss = [df['coord_loss'][df['epoch']==i].mean() for i in range(max(df['epoch']) + 1)]
    fig, ax = plt.subplots()
    plt.plot(epochs, coord_loss)
    ax.tick_params('x', rotation=90)

    plt.title('Coord_Loss_epoch')
    plt.savefig(dest_path+'Coord_Loss_epoch.png')
    plt.show()

    plt.title('Coord_Loss_step')
    plt.plot(df['step'], df['coord_loss'])
    plt.savefig(dest_path+'Coord_Loss_step.png')
    plt.show()

    dist_loss = [df['dist_loss'][df['epoch']==i].mean() for i in range(max(df['epoch']) + 1)]
    fig, ax = plt.subplots()
    plt.plot(epochs, dist_loss)
    ax.tick_params('x', rotation=90)

    plt.title('Dist_Loss_epoch')
    plt.savefig(dest_path+'Dist_Loss_epoch.png')
    plt.show()

    plt.title('Dist_Loss_step')
    plt.plot(df['step'], df['dist_loss'])
    plt.savefig(dest_path+'Dist_Loss_step.png')
    plt.show()

    fig, ax = plt.subplots()

    plt.title('noise_x')
    plt.plot(df['step'], df['real_x'], label='real_noise')
    plt.plot(df['step'], df['pred_x'], label='pred_noise')
    plt.legend()
    plt.savefig(dest_path+'noise_x.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.title('noise_y')
    plt.plot(df['step'], df['real_y'], label='real_noise')
    plt.plot(df['step'], df['pred_y'], label='pred_noise')
    plt.legend()
    plt.savefig(dest_path+'noise_y.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.title('noise_z')
    plt.plot(df['step'], df['real_z'], label='real_noise')
    plt.plot(df['step'], df['pred_z'], label='pred_noise')
    plt.legend()
    plt.savefig(dest_path+'noise_z.png')
    plt.show()

    for i in range(6):
        fig, ax = plt.subplots()
        plt.title(f'Class {i} noises')
        plt.plot(df['step'], df[f'pred_{i}'], label='pred_noise')
        plt.plot(df['step'], df[f'noise_{i}'], label='real_noise')
        plt.legend()
        plt.show()
