"""Small nanotron run with the #272 memory probe attached.

Run from the nanotron repo root, for example:
  CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 /path/to/run_272.py \
      --base-config examples/config_tiny_llama.yaml --dp 2 --tp 1 --accum 3 --out runs/dp2

Model is the tiny llama example with fewer layers and a shorter sequence so it fits on a 24GB card.
Data is nanotron's dummy generator (dataset: null), so nothing to download except the tokenizer.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.getcwd())  # nanotron repo root, for run_train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder, for probe_272


def build_config(args):
    with open(args.base_config) as f:
        cfg = yaml.safe_load(f)

    cfg["general"]["run"] = f"issue272_{os.path.basename(args.out.rstrip('/'))}"
    cfg["general"]["benchmark_csv_path"] = os.path.join(args.out, "bench.csv")
    cfg["profiler"] = None
    cfg["checkpoints"]["checkpoints_path"] = os.path.join(args.out, "ckpt")
    cfg["checkpoints"]["checkpoint_interval"] = 10**6
    cfg["checkpoints"]["save_initial_state"] = False
    cfg["data_stages"][0]["data"]["dataset"] = None

    par = cfg["parallelism"]
    par["dp"] = args.dp
    par["tp"] = args.tp
    par["pp"] = 1
    par["pp_engine"] = args.pp_engine
    par["tp_linear_async_communication"] = args.tp > 1

    cfg["model"]["model_config"]["num_hidden_layers"] = args.layers

    tok = cfg["tokens"]
    tok["micro_batch_size"] = args.mbs
    tok["sequence_length"] = args.seq
    tok["batch_accumulation_per_replica"] = args.accum
    tok["train_steps"] = args.record_steps + 2

    sched = cfg["optimizer"]["learning_rate_scheduler"]
    sched["lr_warmup_steps"] = 1
    sched["lr_decay_steps"] = None

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="examples/config_tiny_llama.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--accum", type=int, default=3)
    # ~570M params: weights, fp32 copies, fp32 grads and Adam states take ~11 GB, fits a 24 GB card.
    # The tied embedding / lm_head (128k x 2048) is kept on purpose: its grad is the biggest one (~500 MB).
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--mbs", type=int, default=2)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--pp-engine", default="1f1b", choices=["1f1b", "afab"])
    p.add_argument("--record-steps", type=int, default=3)
    p.add_argument("--no-probe", action="store_true", help="plain run, to check the probe itself changes nothing")
    args = p.parse_args()

    config_path = build_config(args)

    from nanotron.trainer import DistributedTrainer
    from run_train import get_dataloader

    trainer = DistributedTrainer(config_path)
    dataloader = get_dataloader(trainer)

    if not args.no_probe:
        import probe_272

        probe_272.install(args.out, record_steps=args.record_steps)

    trainer.train(dataloader)


if __name__ == "__main__":
    main()
