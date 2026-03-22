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


def weight_entropy(weights, mask):
    """Compute mean entropy of weight distributions. Higher = more spread out."""
    # weights: (B, L), mask: (B, L)
    w = weights.clamp(min=1e-8)  # avoid log(0)
    entropy = -(w * w.log() * mask.float()).sum(dim=-1)  # (B,)
    return entropy.mean()


def train(config: ESCIConfig, output_dir: str, max_steps: int = None, max_rows: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | fp16: {use_amp}")
    print(f"Weights: {config.use_token_weights} ({config.weight_norm})")
    if config.use_token_weights:
        print(f"Entropy regularization: lambda={config.entropy_lambda}")
        print(f"Softmax temperature: {config.softmax_temperature}")

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

    # Sample queries for periodic weight monitoring
    MONITOR_QUERIES = [
        "men's black leather jacket",
        "women's running shoes size 8",
        "red dress for wedding guest",
    ]
    monitor_encs = None
    if model.weight_head is not None:
        monitor_encs = [
            tokenizer(q, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=config.query_maxlen)
            for q in MONITOR_QUERIES
        ]

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

                # Entropy regularization: penalize collapsed weights
                if weights is not None and config.entropy_lambda > 0:
                    ent = weight_entropy(weights, batch["q_mask"].bool())
                    loss = loss - config.entropy_lambda * ent  # maximize entropy

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

            # Periodic weight monitoring every 500 steps
            if monitor_encs is not None and step % 500 == 0 and step > 0:
                model.eval()
                print(f"\n  [Step {step}] Weight check:")
                with torch.no_grad():
                    for q_text, enc in zip(MONITOR_QUERIES, monitor_encs):
                        enc_d = {k: v.to(device) for k, v in enc.items()}
                        _, mask, hidden = model.encode(enc_d["input_ids"], enc_d["attention_mask"])
                        w = model.weight_head(hidden, mask)[0].cpu()
                        tokens = tokenizer.convert_ids_to_tokens(enc_d["input_ids"][0])
                        m = mask[0].cpu()
                        parts = []
                        for t, wi, mi in zip(tokens, w, m):
                            if mi and t not in ("[CLS]", "[SEP]", "[PAD]"):
                                parts.append(f"{t}={wi:.3f}")
                        print(f"    {q_text[:40]:40s} → {', '.join(parts)}")
                model.train()

            step += 1
            if max_steps and step >= max_steps:
                break

        # Save checkpoint per epoch
        epoch_path = os.path.join(output_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

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
    parser.add_argument("--entropy_lambda", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    args = parser.parse_args()

    config = ESCIConfig(
        use_token_weights=args.use_weights,
        weight_norm=args.norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_lambda=args.entropy_lambda,
        softmax_temperature=args.temperature,
    )
    train(config, args.output_dir, max_steps=args.max_steps, max_rows=args.max_rows)


if __name__ == "__main__":
    main()