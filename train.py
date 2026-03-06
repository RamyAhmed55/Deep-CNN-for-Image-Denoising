# train.py

import os
import torch
import argparse
from tqdm import tqdm
from loss import build_loss
import torch.optim as optim
from config import get_config
from model import build_model
from data import get_dataloader
import torch.backends.cudnn as cudnn
from utils import AverageMeter, seed_everything

cudnn.benchmark = True

# =======================================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="baseline")

    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)

    parser.add_argument("--arch", type=str, default=None, help="DnCNN-S, DnCNN-B, DnCNN-3")
    parser.add_argument("--gaussian_noise_level", type=str, default=None)
    parser.add_argument("--downsampling_factor", type=str, default=None)
    parser.add_argument("--jpeg_quality", type=str, default=None)

    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--task_probs", type=str, default=None)

    parser.add_argument("--resume", action="store_true", help="resume from outputs_dir/checkpoint.pth")

    return parser.parse_args()

# =======================================================================================

def merge_args_into_config(cfg, args):
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if key in cfg and value is not None:
            cfg[key] = value

    if args.resume:
        cfg["resume"] = True

    return cfg

# =======================================================================================

def main():
    args = parse_args()
    cfg = get_config(args.config)
    cfg = merge_args_into_config(cfg, args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.outputs_dir, exist_ok=True)
    seed_everything(cfg["seed"])

    model = build_model(cfg["arch"]).to(device)
    criterion = build_loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["scheduler_milestones"],
        gamma=cfg["scheduler_gamma"]
    )

    ckpt_path = os.path.join(args.outputs_dir, "checkpoint.pth")
    start_epoch = 0
    global_step = 0

    if cfg["resume"] and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        print(f"Resumed from {ckpt_path}: epoch={start_epoch}, step={global_step}")

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        print("Training from random initialization")

    dataset, dataloader = get_dataloader(cfg, args.images_dir)

    print("virtual dataset length:", len(dataset))
    print("iters/epoch:", len(dataloader))

# =======================================================================================

    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        epoch_losses = AverageMeter()

        pbar = tqdm(total=(len(dataset) - len(dataset) % cfg["batch_size"]))
        pbar.set_description(f"epoch: {epoch + 1}/{cfg['num_epochs']}")

        for i, batch in enumerate(dataloader):
            inputs, residual = batch
            inputs = inputs.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)

            pred = model(inputs)
            loss = criterion(pred, residual) / (2 * inputs.size(0))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.update(loss.item(), inputs.size(0))
            global_step += 1

            pbar.set_postfix(
                loss=f"{epoch_losses.avg:.6f}",
                lr=f'{optimizer.param_groups[0]["lr"]:.2e}'
            )
            pbar.update(inputs.size(0))

        pbar.close()
        scheduler.step()

        out_path = os.path.join(args.outputs_dir, f"{cfg['arch']}_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), out_path)
        print("Saved:", out_path)

        torch.save(
            {
                "epoch": epoch + 1,
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print("Saved checkpoint:", ckpt_path)

# =======================================================================================

if __name__ == "__main__":
    main()