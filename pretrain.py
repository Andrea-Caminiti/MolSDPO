import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, EarlyStopping
from data.dataloader import build_qm9_dataloader
from model.model import LightningTabasco


def ema_avg_fn(averaged_model_parameter, current_model_parameter):
    decay = 0.999
    return decay * averaged_model_parameter + (1 - decay) * current_model_parameter


def pretrain(args):
    torch.set_float32_matmul_precision('high')
    module, vocab_enc2atom, vocab_atom2enc = build_qm9_dataloader(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    vocab_enc2atom = vocab_enc2atom.to(args.device)
    args.vocab_size = len(vocab_enc2atom)

    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/Pretrain/ckpts/",
        save_top_k=5,
        monitor="val/denoise_loss_t100",
        mode='min',
        filename='{epoch}-{step}-{loss:.2f}'
    )

    EMA = StochasticWeightAveraging(1e-3, avg_fn=ema_avg_fn)

    # EarlyStopping monitors val/denoise_loss_t100 which is logged in
    # validation_step — no separate callback needed.
    early_stop = EarlyStopping(
        monitor='val/denoise_loss_t100',
        mode='min',
        patience=8,
        min_delta=0.001,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision="32",
        max_steps=args.max_steps,
        enable_progress_bar=True,
        logger=CSVLogger("logs", name="Pretrain"),
        log_every_n_steps=50,
        callbacks=[checkpoint_callback, early_stop],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        val_check_interval=2500,
    )

    model = LightningTabasco(args, vocab_enc2atom)
    model.model = torch.compile(model.model)
    trainer.fit(model=model, datamodule=module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qm9')
    parser.add_argument('--data-root', default='data/QM9')
    parser.add_argument('--max_steps', type=int, default=1_000_000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--sample-steps', type=int, default=25)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    os.makedirs("logs/Pretrain/ckpts/", exist_ok=True)
    pretrain(args)