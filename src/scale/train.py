from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import re

import torch
from ema_pytorch import EMA
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from scale.dataset import move_batch
from scale.losses import compute_losses
from scale.runtime import (
    apply_overrides,
    append_jsonl,
    cosine_schedule,
    create_dataset,
    create_loader,
    evaluate_loader,
    load_config,
    load_model,
    save_checkpoint,
    seed_everything,
)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    last_path = output_dir / "last.pt"
    if last_path.is_file():
        return last_path
    numbered = []
    for path in output_dir.glob("step-*.pt"):
        match = re.fullmatch(r"step-(\d+)\.pt", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    return max(numbered, key=lambda item: item[0])[1] if numbered else None


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    apply_overrides(config, args.override)
    seed_everything(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_dataset = create_dataset(config, "train")
    eval_dataset = create_dataset(config, "eval")
    train_loader = create_loader(train_dataset, config, "train")
    eval_loader = create_loader(eval_dataset, config, "eval")
    model, _ = load_model(config, None, device)
    encoder_parameters = []
    main_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("lyrics_model.encoder."):
            encoder_parameters.append(parameter)
        else:
            main_parameters.append(parameter)
    parameter_groups = [
        {"params": main_parameters, "lr": float(config.training.lr)}
    ]
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": float(config.training.lyrics_encoder_lr),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(config.training.lr),
        betas=tuple(config.training.betas),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = cosine_schedule(
        optimizer, int(config.training.warmup_steps), int(config.training.max_steps)
    )
    ema = EMA(
        model,
        beta=float(config.training.ema_beta),
        update_after_step=int(config.training.ema_after_step),
        update_every=1,
        include_online_model=False,
    )
    output_dir = Path(args.output_dir or config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_score = -1.0
    best_acc = -1.0
    best_hr05f = -1.0
    if args.resume and args.no_auto_resume:
        raise ValueError("Use either --resume or --no-auto-resume, not both")
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and not args.no_auto_resume:
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path is None:
            logger.info("No checkpoint found in {}; starting from scratch", output_dir)
        else:
            logger.info("Automatically resuming from {}", resume_path)
    if resume_path:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        step = int(checkpoint["step"])
        best_score = float(checkpoint.get("best_score", -1.0))
        best_acc = float(checkpoint.get("best_acc", -1.0))
        best_hr05f = float(checkpoint.get("best_hr05f", -1.0))
    accumulation = int(config.training.accumulation_steps)
    amp_dtype = torch.bfloat16 if str(config.training.amp_dtype) == "bfloat16" else torch.float16
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=int(config.training.max_steps), initial=step, desc="train")
    while step < int(config.training.max_steps):
        for batch in train_loader:
            if batch is None:
                continue
            batch = move_batch(batch, device)
            use_amp = bool(config.training.amp) and device.type == "cuda"
            autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with autocast:
                outputs = model(batch)
                losses = compute_losses(outputs, batch, config.loss)
                loss = losses["loss"] / accumulation
            loss.backward()
            micro_step = getattr(main, "micro_step", 0) + 1
            main.micro_step = micro_step
            if micro_step % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.grad_clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            step += 1
            progress.update(1)
            if step % int(config.training.log_interval) == 0:
                record = {"step": step, "lr": scheduler.get_last_lr()[0]}
                record.update({name: float(value.item()) for name, value in losses.items()})
                append_jsonl(output_dir / "train.jsonl", record)
                progress.set_postfix(loss=f"{record['loss']:.4f}")
            should_evaluate = step % int(config.training.eval_interval) == 0
            should_save = step % int(config.training.save_interval) == 0
            metrics = None
            if should_evaluate:
                metrics = evaluate_loader(ema.ema_model, eval_loader, config, device)
                score = (metrics["HR0.5F"] + metrics["HR3F"] + metrics["ACC"]) / 3.0
                append_jsonl(output_dir / "eval.jsonl", {"step": step, **metrics, "score": score})
                model.train()
                improved_score = score > best_score
                improved_acc = metrics["ACC"] > best_acc
                improved_hr05f = metrics["HR0.5F"] > best_hr05f
                best_score = max(best_score, score)
                best_acc = max(best_acc, metrics["ACC"])
                best_hr05f = max(best_hr05f, metrics["HR0.5F"])
                eval_payload = {
                    "model": model.state_dict(),
                    "ema_model": ema.ema_model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_score": best_score,
                    "best_acc": best_acc,
                    "best_hr05f": best_hr05f,
                    "config": OmegaConf.to_container(config, resolve=True),
                }
                if improved_score:
                    save_checkpoint(
                        output_dir / "best.pt",
                        eval_payload,
                    )
                if improved_acc:
                    save_checkpoint(output_dir / "best_acc.pt", eval_payload)
                if improved_hr05f:
                    save_checkpoint(output_dir / "best_hr05f.pt", eval_payload)
            if should_save:
                payload = {
                    "model": model.state_dict(),
                    "ema_model": ema.ema_model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_score": best_score,
                    "best_acc": best_acc,
                    "best_hr05f": best_hr05f,
                    "config": OmegaConf.to_container(config, resolve=True),
                }
                save_checkpoint(output_dir / f"step-{step}.pt", payload)
                save_checkpoint(output_dir / "last.pt", payload)
            if step >= int(config.training.max_steps):
                break
    progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    main(parser.parse_args())
