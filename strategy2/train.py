"""Strategy 2 training: query encoder + weight head, doc encoder frozen."""
import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import S2Config
from .model import ColBERTWeightedS2
from colbert_weighted.data import MSMARCOTriplesDataset, ColBERTCollator
from torch.utils.data import DataLoader


def pairwise_softmax_loss(pos_scores, neg_scores):
    scores = torch.stack([pos_scores, neg_scores], dim=-1)
    return -F.log_softmax(scores, dim=-1)[:, 0].mean()


def train(config: S2Config, output_dir: str, max_steps: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Strategy 2: query encoder + weight head train, doc frozen")
    print(f"Weight norm: {config.weight_norm}")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = ColBERTWeightedS2(config).to(device)

    # Load train data, excluding held-out eval queries
    full_ds = MSMARCOTriplesDataset(split="train")
    train_size = len(full_ds) - config.eval_holdout
    train_ds = torch.utils.data.Subset(full_ds, range(train_size))
    print(f"Train: {train_size} queries, Held-out eval: {config.eval_holdout}")

    collator = ColBERTCollator(tokenizer, config.query_maxlen, config.doc_maxlen)
    loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    # Two param groups: encoder (low LR) + weight head (high LR)
    encoder_params = list(model.bert.parameters()) + list(model.linear.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config.encoder_lr},
        {"params": model.weight_head.parameters(), "lr": config.weight_head_lr},
    ])

    total_steps = max_steps or (len(loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    n_encoder = sum(p.numel() for p in encoder_params)
    n_head = sum(p.numel() for p in model.weight_head.parameters())
    print(f"Trainable: encoder={n_encoder:,} + weight_head={n_head} = {n_encoder+n_head:,}")

    os.makedirs(output_dir, exist_ok=True)
    log = []
    step = 0

    model.train()
    for epoch in range(config.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_scores, neg_scores, weights = model(**batch)
            loss = pairwise_softmax_loss(pos_scores, neg_scores)

            loss.backward()
            if (step + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(model.weight_head.state_dict(), os.path.join(output_dir, "weight_head.pt"))
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"Saved to {output_dir}")
    return model
