"""ESCI gap-based weight training.

Step 1: Compute per-token MaxSim gaps between Exact and Substitute products.
Step 2: Train weight head to predict these gaps from query hidden states alone.

Encoder is frozen — only the weight head (769 params) trains.
"""
import json
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import ESCIConfig
from .model import ColBERTESCI
from .data import get_dataloader
from colbert_weighted.scoring import maxsim


def compute_token_gaps(Q, D_pos, D_neg, q_mask, d_pos_mask, d_neg_mask, temperature=0.1):
    """Compute per-token MaxSim gap: how much better each query token matches pos vs neg.

    Returns: (B, L) normalized gap weights (softmax over gaps).
    """
    # Per-token MaxSim to positive doc
    sim_pos = torch.bmm(Q, D_pos.transpose(1, 2))  # (B, Lq, Ld_pos)
    sim_pos = sim_pos.masked_fill(~d_pos_mask.unsqueeze(1), float("-inf"))
    max_sim_pos, _ = sim_pos.max(dim=-1)  # (B, Lq)

    # Per-token MaxSim to negative doc
    sim_neg = torch.bmm(Q, D_neg.transpose(1, 2))  # (B, Lq, Ld_neg)
    sim_neg = sim_neg.masked_fill(~d_neg_mask.unsqueeze(1), float("-inf"))
    max_sim_neg, _ = sim_neg.max(dim=-1)  # (B, Lq)

    # Gap = how much better this token matches pos vs neg
    gaps = (max_sim_pos - max_sim_neg)  # (B, Lq)
    gaps = gaps * q_mask.float()  # zero out padding

    # Normalize to get target weights (softmax with temperature for sharper targets)
    gaps = gaps.masked_fill(~q_mask, float("-inf"))
    target_weights = F.softmax(gaps / temperature, dim=-1)  # lower temp = sharper targets

    return target_weights, gaps


def train_gap(config: ESCIConfig, encoder_path: str, output_dir: str,
              max_steps: int = None, max_rows: int = None):
    """Train weight head on gap-derived targets. Encoder frozen."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | fp16: {use_amp}")
    print("Gap-based weight training: encoder frozen, weight head learns token importance")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    # Load frozen encoder (baseline or original ColBERTv2)
    model = ColBERTESCI(config).to(device)
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading encoder from {encoder_path}")
        state = torch.load(encoder_path, map_location=device)
        # Load only encoder weights, skip weight_head
        encoder_keys = {k: v for k, v in state.items()
                        if not k.startswith("weight_head")}
        model.load_state_dict(encoder_keys, strict=False)
    else:
        print("Using original ColBERTv2 encoder (no fine-tuning)")

    # Freeze encoder
    for p in model.bert.parameters():
        p.requires_grad = False
    for p in model.linear.parameters():
        p.requires_grad = False

    # Only weight head trains
    optimizer = torch.optim.AdamW(model.weight_head.parameters(), lr=config.weight_head_lr)
    scaler = GradScaler(enabled=use_amp)

    loader = get_dataloader(config, tokenizer, split="train", max_rows=max_rows)
    total_steps = max_steps or (len(loader) * config.epochs)
    print(f"Weight head params: {sum(p.numel() for p in model.weight_head.parameters())}")
    print(f"Steps: {total_steps}")

    # Sample queries for monitoring
    MONITOR_QUERIES = [
        "men's black leather jacket",
        "women's running shoes size 8",
        "red dress for wedding guest",
    ]
    monitor_encs = [
        tokenizer(q, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=config.query_maxlen)
        for q in MONITOR_QUERIES
    ]

    os.makedirs(output_dir, exist_ok=True)
    log = []
    step = 0

    model.weight_head.train()
    for epoch in range(config.epochs):
        if max_steps and step >= max_steps:
            break
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                Q, q_mask, q_hidden = model.encode(batch["q_ids"], batch["q_mask"])
                D_pos, d_pos_mask, _ = model.encode(batch["d_pos_ids"], batch["d_pos_mask"])
                D_neg, d_neg_mask, _ = model.encode(batch["d_neg_ids"], batch["d_neg_mask"])

                # Compute target weights from MaxSim gaps
                target_weights, gaps = compute_token_gaps(
                    Q, D_pos, D_neg, q_mask.bool(), d_pos_mask.bool(), d_neg_mask.bool(),
                    temperature=config.softmax_temperature)

            # Forward through weight head (only this has gradients)
            q_hidden = q_hidden.detach().requires_grad_(True)
            with autocast(enabled=use_amp):
                predicted_weights = model.weight_head(q_hidden, q_mask.bool())
                loss = F.mse_loss(predicted_weights, target_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            log.append({"step": step, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            # Monitor weights + mini eval every 500 steps
            if step % 500 == 0 and step > 0:
                model.weight_head.eval()
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

                # Mini eval on 10 queries
                from .evaluate import evaluate
                mini_results = evaluate(model, config, device, max_queries=10, split="test")
                v_mrr = mini_results["vanilla_mrr@10"]
                w_mrr = mini_results["weighted_mrr@10"]
                v_attr = mini_results.get("vanilla_attr_mrr@10", 0) or 0
                w_attr = mini_results.get("weighted_attr_mrr@10", 0) or 0
                print(f"  Mini eval: vanilla={v_mrr:.4f} weighted={w_mrr:.4f} "
                      f"v_attr={v_attr:.4f} w_attr={w_attr:.4f}")
                model.weight_head.train()

            step += 1
            if max_steps and step >= max_steps:
                break

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(model.weight_head.state_dict(), os.path.join(output_dir, "weight_head.pt"))
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"Saved to {output_dir}")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESCI gap-based weight training")
    parser.add_argument("--encoder_path", default=None,
                        help="Path to trained encoder model.pt (default: use original ColBERTv2)")
    parser.add_argument("--output_dir", default="outputs/esci/gap_weighted")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--gap_temperature", type=float, default=0.1)
    args = parser.parse_args()

    config = ESCIConfig(
        use_token_weights=True,
        weight_norm="softmax",
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_lambda=0.0,
        softmax_temperature=args.gap_temperature,
    )

    model = train_gap(config, args.encoder_path, args.output_dir,
                      max_steps=args.max_steps, max_rows=args.max_rows)

    # Quick eval
    from .evaluate import evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results = evaluate(model, config, device, max_queries=args.num_eval, split="test")

    print(f"\n  Vanilla  MRR@10: {results['vanilla_mrr@10']:.4f}")
    print(f"  Weighted MRR@10: {results['weighted_mrr@10']:.4f}")
    if results.get("vanilla_attr_mrr@10"):
        print(f"  Vanilla  Attr MRR@10: {results['vanilla_attr_mrr@10']:.4f}")
        print(f"  Weighted Attr MRR@10: {results['weighted_attr_mrr@10']:.4f}")
    print(f"  Queries: {results['num_queries']}")


if __name__ == "__main__":
    main()
