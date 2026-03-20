"""Training loop: frozen ColBERTv2 encoder, only weight head trains."""
import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import ExpConfig
from .model import ColBERTWeighted
from .data import get_dataloader


def pairwise_softmax_loss(pos_scores, neg_scores):
    """Standard pairwise ranking loss."""
    scores = torch.stack([pos_scores, neg_scores], dim=-1)  # (B, 2)
    return -F.log_softmax(scores, dim=-1)[:, 0].mean()


def train(config: ExpConfig, output_dir: str, max_steps: int = None, max_rows: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Token weights: {config.use_token_weights} ({config.weight_norm})")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = ColBERTWeighted(config).to(device)
    loader = get_dataloader(config, tokenizer, split="train", max_rows=max_rows)

    # Only weight head is trainable
    trainable = list(model.weight_head.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_params} (weight head only)")

    optimizer = torch.optim.AdamW(trainable, lr=config.weight_head_lr)
    total_steps = max_steps or (len(loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps,
    )

    os.makedirs(output_dir, exist_ok=True)
    log = []
    step = 0

    model.train()  # only affects weight head (dropout/batchnorm); encoder is frozen
    for epoch in range(config.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_scores, neg_scores, weights = model(**batch)
            loss = pairwise_softmax_loss(pos_scores, neg_scores)

            loss.backward()
            if (step + 1) % config.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            log.append({"step": step, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # Save only the weight head (129 params) — encoder is unchanged ColBERTv2
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.weight_head.state_dict(), os.path.join(output_dir, "weight_head.pt"))
    # Also save full state dict for convenience
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"Saved to {output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--output_dir", default="outputs/weighted")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    config = ExpConfig(
        use_token_weights=True,
        weight_norm=args.norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    train(config, args.output_dir, max_steps=args.max_steps, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
