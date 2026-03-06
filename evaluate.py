import os
import glob
import argparse
import cv2
import torch
import numpy as np
from metrics import calculate_psnr, calculate_ssim, summarize_metrics
from model import DnCNN

# =======================================================================================

def bgr_to_y01(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0
    return y


def bicubic_down_up_y(y01: np.ndarray, scale: int) -> np.ndarray:
    h, w = y01.shape
    y255 = (y01 * 255.0).astype(np.float32)

    small_w = max(1, int(round(w / scale)))
    small_h = max(1, int(round(h / scale)))

    small = cv2.resize(y255, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    return np.clip(up / 255.0, 0.0, 1.0)


def jpeg_compress_y(y01: np.ndarray, quality: int) -> np.ndarray:
    y255 = np.clip(y01 * 255.0, 0, 255).astype(np.uint8)

    ok, enc = cv2.imencode(".jpg", y255, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return np.clip(dec.astype(np.float32) / 255.0, 0.0, 1.0)


def load_model(weights_path: str, device: torch.device, num_layers: int):
    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # strip module.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model = DnCNN(num_layers=num_layers).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("missing:", len(missing), "unexpected:", len(unexpected))
    if len(missing) > 0:
        print("missing sample:", missing[:5])
    if len(unexpected) > 0:
        print("unexpected sample:", unexpected[:5])

    model.eval()
    return model


def collect_files(dataset_dir: str):
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        files.extend(glob.glob(os.path.join(dataset_dir, ext)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, required=True, help="folder of clean HR images")
    parser.add_argument("--weights", type=str, required=True, help="path to model weights")
    parser.add_argument("--task", type=str, required=True, choices=["gaussian", "sr", "jpeg"])
    parser.add_argument("--model_output", type=str, default="denoised", choices=["denoised", "residual"],
                        help="set residual if model predicts noise/residual; set denoised if model returns clean image directly")
    parser.add_argument("--num_layers", type=int, default=20)

    # task params
    parser.add_argument("--sigma", type=float, default=25.0, help="for gaussian")
    parser.add_argument("--scale", type=int, default=3, help="for sr")
    parser.add_argument("--quality", type=int, default=40, help="for jpeg")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

# =======================================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    files = collect_files(args.dataset_dir)
    print("images found:", len(files))
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {args.dataset_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(args.weights, device, args.num_layers)

    psnr_list = []
    ssim_list = []

# =======================================================================================

    with torch.no_grad():
        for f in files:
            img_bgr = cv2.imread(f)
            if img_bgr is None:
                continue

            clean = bgr_to_y01(img_bgr)

            if args.task == "gaussian":
                noise = np.random.randn(*clean.shape).astype(np.float32) * (args.sigma / 255.0)
                inp = np.clip(clean + noise, 0.0, 1.0)

            elif args.task == "sr":
                inp = bicubic_down_up_y(clean, args.scale)

            else:  # jpeg
                inp = jpeg_compress_y(clean, args.quality)

            inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)

            pred = model(inp_t)

            if args.model_output == "residual":
                out_t = inp_t - pred
            else:
                out_t = pred

            out_t = torch.clamp(out_t, 0.0, 1.0)
            out = out_t.squeeze().cpu().numpy()

            psnr_val = calculate_psnr(out, clean, data_range=1.0)
            ssim_val = calculate_ssim(out, clean, data_range=1.0)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            print(f"{os.path.basename(f)}  PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}")

    avg_psnr, avg_ssim = summarize_metrics(psnr_list, ssim_list)

    print("\n" + "=" * 60)
    print(f"TASK: {args.task}")
    if args.task == "gaussian":
        print(f"sigma = {args.sigma}")
    elif args.task == "sr":
        print(f"scale = x{args.scale}")
    else:
        print(f"jpeg quality = {args.quality}")

    print(f"Average PSNR = {avg_psnr:.2f} dB")
    print(f"Average SSIM = {avg_ssim:.4f}")
    print("=" * 60)

# =======================================================================================

if __name__ == "__main__":
    main()