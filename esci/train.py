"""ESCI training: full encoder + optional weight head, fp16."""
import json
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import ESCIConfig
from .model import ColBERTESCI
from .data import get_dataloader


def pairwise_softmax_loss(pos_scores, neg_scores):
    scores = torch.stack([pos_scores, neg_scores], dim=-1)
    return -F.log_softmax(scores, dim=-1)[:, 0].mean()


def train(config: ESCIConfig, output_dir: str, max_steps: int = None, max_rows: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | fp16: {use_amp}")
    print(f"Weights: {config.use_token_weights} ({config.weight_norm})")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = ColBERTESCI(config).to(device)
    loader = get_dataloader(config, tokenizer, split="train", max_rows=max_rows)

    # Param groups
    params = [{"params": list(model.bert.parameters()) + list(model.linear.parameters()),
               "lr": config.encoder_lr}]
    if model.weight_head is not None:
        params.append({"params": model.weight_head.parameters(), "lr": config.weight_head_lr})

    optimizer = torch.optim.AdamW(params)
    total_steps = max_steps or (len(loader) * config.epochs)
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = GradScaler(enabled=use_amp)

    n_params = sum(p.numel() for pg in params for p in pg["params"])
    print(f"Trainable: {n_params:,} params | Steps: {total_steps}")

    os.makedirs(output_dir, exist_ok=True)
    log = []
    step = 0

    model.train()
    for epoch in range(config.epochs):
        if max_steps and step >= max_steps:
            break
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=use_amp):
                pos_scores, neg_scores, weights = model(**batch)
                loss = pairwise_softmax_loss(pos_scores, neg_scores)

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            log.append({"step": step, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1
            if max_steps and step >= max_steps:
                break

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    if model.weight_head is not None:
        torch.save(model.weight_head.state_dict(), os.path.join(output_dir, "weight_head.pt"))
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"Saved to {output_dir}")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESCI ColBERT training")
    parser.add_argument("--output_dir", default="outputs/esci")
    parser.add_argument("--use_weights", action="store_true")
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    config = ESCIConfig(
        use_token_weights=args.use_weights,
        weight_norm=args.norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    train(config, args.output_dir, max_steps=args.max_steps, max_rows=args.max_rows)


if __name__ == "__main__":
    main()