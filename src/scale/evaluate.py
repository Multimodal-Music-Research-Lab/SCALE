from __future__ import annotations

import argparse
from pathlib import Path

import torch
from scale.metrics import write_evaluation_outputs, write_predictions
from scale.runtime import (
    apply_overrides,
    create_dataset,
    create_loader,
    evaluate_loader_detailed,
    load_config,
    load_model,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-dir")
    parser.add_argument("--est-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--online-model", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.override)
    device = torch.device(args.device)
    dataset = create_dataset(config, "eval")
    loader = create_loader(dataset, config, "eval")
    model, _ = load_model(config, args.checkpoint, device, prefer_ema=not args.online_model)
    result = evaluate_loader_detailed(model, loader, config, device)
    output_dir = Path(args.output_dir)
    table = write_evaluation_outputs(
        output_dir,
        result["records"],
        result["summary"],
        result["confusion"],
    )
    write_predictions(result["predictions"], args.prediction_dir, args.est_dir)
    print(table)
    print(f"\nDetailed evaluation written to: {output_dir}")
