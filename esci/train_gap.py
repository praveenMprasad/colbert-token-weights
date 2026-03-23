"""ESCI gap-based weight training with multi-negative averaged targets.

Compute per-token MaxSim gaps between Exact and ALL Substitute products,
average across negatives for stable importance targets.
Train weight head (frozen encoder) with MSE loss to predict these targets.

Multi-stage eval: full eval every N steps to track progress.
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
from .data import get_multi_neg_dataloader
from colbert_weighted.scoring import maxsim


def compute_multi_neg_gaps(Q, D_pos, d_pos_mask, D_neg_all, d_neg_mask_all,
                           neg_counts, q_mask, temperature=0.3):
    """Compute per-token MaxSim gap averaged across multiple negatives.

    Args:
        Q: (B, Lq, dim)
        D_pos: (B, Ld, dim)
        d_pos_mask: (B, Ld)
        D_neg_all: (N_total, Ld, dim) — all negatives flattened
        d_neg_mask_all: (N_total, Ld)
        neg_counts: (B,) — how many negatives per query
        q_mask: (B, Lq)
        temperature: softmax temperature for target sharpness

    Returns:
        target_weights: (B, Lq) averaged gap-based importance weights
    """
    B, Lq, dim = Q.shape

    # Per-token MaxSim to positive
    sim_pos = torch.bmm(Q, D_pos.transpose(1, 2))
    sim_pos = sim_pos.masked_fill(~d_pos_mask.unsqueeze(1), float("-inf"))
    max_sim_pos, _ = sim_pos.max(dim=-1)  # (B, Lq)

    # Average gaps across all negatives per query
    avg_gaps = torch.zeros(B, Lq, device=Q.device)
    neg_offset = 0
    for i in range(B):
        n = neg_counts[i].item()
        if n == 0:
            continue
        # Get this query's negatives
        D_negs_i = D_neg_all[neg_offset:neg_offset + n]  # (n, Ld, dim)
        d_neg_mask_i = d_neg_mask_all[neg_offset:neg_offset + n]  # (n, Ld)

        # Expand query for all negatives: (n, Lq, dim)
        Q_i = Q[i].unsqueeze(0).expand(n, -1, -1)

        # Per-token MaxSim to each negative
        sim_neg = torch.bmm(Q_i, D_negs_i.transpose(1, 2))  # (n, Lq, Ld)
        sim_neg = sim_neg.masked_fill(~d_neg_mask_i.unsqueeze(1), float("-inf"))
        max_sim_neg, _ = sim_neg.max(dim=-1)  # (n, Lq)

        # Gap per negative, then average
        gaps_i = max_sim_pos[i].unsqueeze(0) - max_sim_neg  # (n, Lq)
        avg_gaps[i] = gaps_i.mean(dim=0)  # (Lq,)

        neg_offset += n

    # Mask and normalize
    avg_gaps = avg_gaps * q_mask.float()
    avg_gaps = avg_gaps.masked_fill(~q_mask, float("-inf"))
    target_weights = F.softmax(avg_gaps / temperature, dim=-1)

    return target_weights, avg_gaps


def train_gap(config: ESCIConfig, encoder_path: str, output_dir: str,
              max_steps: int = None, max_rows: int = None,
              eval_every: int = 500, num_eval: int = 50, max_negs: int = 8):
    """Train weight head on multi-negative gap targets. Encoder frozen.

    Multi-stage eval: runs full eval every `eval_every` steps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | fp16: {use_amp}")
    print(f"Multi-neg gap training: max_negs={max_negs}, temp={config.softmax_temperature}")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    # Load frozen encoder
    model = ColBERTESCI(config).to(device)
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading encoder from {encoder_path}")
        state = torch.load(encoder_path, map_location=device)
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

    optimizer = torch.optim.AdamW(model.weight_head.parameters(), lr=config.weight_head_lr)
    scaler = GradScaler(enabled=use_amp)

    loader = get_multi_neg_dataloader(config, tokenizer, split="train",
                                       max_rows=max_rows, max_negs=max_negs)
    total_steps = max_steps or (len(loader) * config.epochs)
    print(f"Weight head params: {sum(p.numel() for p in model.weight_head.parameters())}")
    print(f"Steps: {total_steps} | Eval every: {eval_every} steps")

    # Monitor queries
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
    eval_log = []
    step = 0

    model.weight_head.train()
    for epoch in range(config.epochs):
        if max_steps and step >= max_steps:
            break
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move tensors to device
            q_ids = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            d_pos_ids = batch["d_pos_ids"].to(device)
            d_pos_mask = batch["d_pos_mask"].to(device)
            d_neg_ids = batch["d_neg_ids"].to(device)
            d_neg_mask = batch["d_neg_mask"].to(device)
            neg_counts = batch["neg_counts"].to(device)

            with torch.no_grad():
                Q, q_m, q_hidden = model.encode(q_ids, q_mask)
                D_pos, d_pos_m, _ = model.encode(d_pos_ids, d_pos_mask)
                D_neg_all, d_neg_m_all, _ = model.encode(d_neg_ids, d_neg_mask)

                target_weights, gaps = compute_multi_neg_gaps(
                    Q, D_pos, d_pos_m, D_neg_all, d_neg_m_all,
                    neg_counts, q_m.bool(),
                    temperature=config.softmax_temperature)

            # Forward through weight head
            q_hidden = q_hidden.detach().requires_grad_(True)
            with autocast(enabled=use_amp):
                predicted_weights = model.weight_head(q_hidden, q_m.bool())
                loss = F.mse_loss(predicted_weights, target_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            log.append({"step": step, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            # Multi-stage eval
            if step > 0 and step % eval_every == 0:
                _run_stage_eval(model, config, tokenizer, device,
                                MONITOR_QUERIES, monitor_encs,
                                step, num_eval, eval_log, output_dir)

            step += 1
            if max_steps and step >= max_steps:
                break

    # Final save
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(model.weight_head.state_dict(), os.path.join(output_dir, "weight_head.pt"))
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(output_dir, "eval_log.json"), "w") as f:
        json.dump(eval_log, f)
    print(f"Saved to {output_dir}")

    # Final eval
    _run_stage_eval(model, config, tokenizer, device,
                    MONITOR_QUERIES, monitor_encs,
                    step, num_eval, eval_log, output_dir, final=True)

    return model


def _run_stage_eval(model, config, tokenizer, device,
                    monitor_queries, monitor_encs,
                    step, num_eval, eval_log, output_dir, final=False):
    """Run weight inspection + full eval at a training checkpoint."""
    from .evaluate import evaluate

    label = "FINAL" if final else f"Step {step}"
    model.weight_head.eval()
    print(f"\n{'='*50}")
    print(f"  [{label}] Weight check + eval")
    print(f"{'='*50}")

    # Weight inspection
    with torch.no_grad():
        for q_text, enc in zip(monitor_queries, monitor_encs):
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

    # Full eval
    results = evaluate(model, config, device, max_queries=num_eval, split="test")
    v_mrr = results["vanilla_mrr@10"]
    w_mrr = results["weighted_mrr@10"]
    v_attr = results.get("vanilla_attr_mrr@10") or 0
    w_attr = results.get("weighted_attr_mrr@10") or 0
    v_sep = results.get("vanilla_separation") or 0
    w_sep = results.get("weighted_separation") or 0

    print(f"  Vanilla  MRR@10={v_mrr:.4f}  Attr={v_attr:.4f}  E>S={v_sep:.4f}")
    print(f"  Weighted MRR@10={w_mrr:.4f}  Attr={w_attr:.4f}  E>S={w_sep:.4f}")
    print(f"  Delta MRR: {w_mrr - v_mrr:+.4f}  Delta Attr: {w_attr - v_attr:+.4f}")

    eval_log.append({
        "step": step,
        "vanilla_mrr": v_mrr, "weighted_mrr": w_mrr,
        "vanilla_attr": v_attr, "weighted_attr": w_attr,
        "vanilla_sep": v_sep, "weighted_sep": w_sep,
        "delta_mrr": w_mrr - v_mrr,
    })

    # Save checkpoint at each eval stage
    ckpt_path = os.path.join(output_dir, f"weight_head_step{step}.pt")
    torch.save(model.weight_head.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    model.weight_head.train()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESCI multi-neg gap weight training")
    parser.add_argument("--encoder_path", default=None)
    parser.add_argument("--output_dir", default="outputs/esci/gap_multineg")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--gap_temperature", type=float, default=0.3)
    parser.add_argument("--max_negs", type=int, default=8)
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
                      max_steps=args.max_steps, max_rows=args.max_rows,
                      eval_every=args.eval_every, num_eval=args.num_eval,
                      max_negs=args.max_negs)


if __name__ == "__main__":
    main()
