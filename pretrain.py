import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback, EarlyStopping
from data.dataloader import build_qm9_dataloader
from model.model import LightningTabasco
from pretrain_validation import DiffusionValidationCallback, EarlyStoppingOnValidation

class Grads(Callback):
    def on_after_backward(self, a, b) -> None:
        print("on_after_backward enter")
        try:
            for name, param in a.named_parameters():
                if param.grad is None:
                    print(name)
        except:
            pass
        try:
            for name, param in b.named_parameters():
                if param.grad is None:
                    print(name)
        except:
            pass
        print("on_after_backward exit")

def ema_avg_fn(averaged_model_parameter, current_model_parameter):
    # classic EMA formula: v = decay*v + (1-decay)*x
    decay = 0.999  # tune this later
    return decay * averaged_model_parameter + (1 - decay) * current_model_parameter

def pretrain(args):
    torch.set_float32_matmul_precision('medium')

    module, vocab_enc2atom, vocab_atom2encab = build_qm9_dataloader(root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    vocab_enc2atom = vocab_enc2atom.to(args.device)
    args.vocab_size = len(vocab_enc2atom)

    checkpoint_callback = ModelCheckpoint(dirpath="logs/Pretrain/ckpts/", save_top_k=5, monitor="val/denoise_loss_t100", mode='min', filename='{epoch}-{step}-{loss:.2f}')
    EMA = StochasticWeightAveraging(1e-3, avg_fn=ema_avg_fn)

    trainer = pl.Trainer(accelerator='gpu', 
                         devices=1, 
                         precision="16-mixed", 
                         max_steps=args.max_steps,
                         enable_progress_bar=True, 
                         logger=CSVLogger("logs", name="Pretrain"), 
                         log_every_n_steps=920, 
                         callbacks=[checkpoint_callback, EMA],
                         gradient_clip_val=1.0, 
                         gradient_clip_algorithm="norm")
    
    
    model = LightningTabasco(args, vocab_enc2atom)  # ← was just (args)
    model.model = torch.compile(model.model)
    trainer.fit(model=model, datamodule=module) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qm9')
    parser.add_argument('--data-root', default='data/QM9')
    parser.add_argument('--max_steps', type=int, default=1_000_000)
    parser.add_argument('--batch-size', type=int, default=128)
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
    os.makedirs("logs/Confronto_Modelli_SDPO/ckpts/", exist_ok=True)
    pretrain(args)
