import os
import glob
import random
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader


# =======================================================================================

class Dataset(object):
    """
    Paper-like DnCNN-3 dataset (mixture of tasks, one per sample):
      (1) Gaussian denoising: sigma in [0,55]
      (2) SISR artifact removal: bicubic down + bicubic up with scale in {2,3,4}
      (3) JPEG deblocking: quality in [5,99]

    Returns:
      input:  CHW float32 in [0,1]
      label:  CHW float32 in [0,1] (clean)
    """

    def __init__(
        self,
        images_dir,
        patch_size=50,
        sigma_range=(0, 55),
        sr_scales=(2, 3, 4),
        jpeg_quality_range=(5, 99),
        task_probs=(1/3, 1/3, 1/3),
        rgb=False,              # False = grayscale/L (close to Y);
        add_augment=True,
        steps_per_epoch=50, 
        extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        seed=123,
    ):
        self.images_dir = images_dir
        self.patch_size = int(patch_size)
        self.sigma_range = sigma_range
        self.sr_scales = tuple(sr_scales)
        self.jpeg_quality_range = jpeg_quality_range
        self.task_probs = task_probs
        self.rgb = rgb
        self.add_augment = add_augment
        self.steps_per_epoch = int(steps_per_epoch)
        self.extensions = tuple(e.lower() for e in extensions)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # collect image files
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.image_files = [p for p in self.image_files if p.lower().endswith(self.extensions)]

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        # sanity
        if not (isinstance(self.task_probs, (list, tuple)) and len(self.task_probs) == 3):
            raise ValueError("task_probs must be a 3-tuple/list: (gauss, sr, jpeg)")
        if sum(self.task_probs) <= 0:
            raise ValueError("task_probs sum must be > 0")

    def __len__(self):
        return self.steps_per_epoch * 128
    
# =======================================================================================

    @staticmethod
    def _augment(img: np.ndarray, mode: int) -> np.ndarray:
        # img: HWC
        if mode == 0:
            return img
        if mode == 1:
            return np.flipud(img)
        if mode == 2:
            return np.rot90(img)
        if mode == 3:
            return np.flipud(np.rot90(img))
        if mode == 4:
            return np.rot90(img, k=2)
        if mode == 5:
            return np.flipud(np.rot90(img, k=2))
        if mode == 6:
            return np.rot90(img, k=3)
        if mode == 7:
            return np.flipud(np.rot90(img, k=3))
        raise ValueError("mode should be 0..7")

    def _read_image(self, path: str) -> np.ndarray:
        img = Image.open(path)
        img = img.convert("RGB" if self.rgb else "L")
        arr = np.array(img).astype(np.float32) 
        if arr.ndim == 2:
            arr = arr[..., None] 
        return arr
    
# =======================================================================================


    def __getitem__(self, idx):
    
        img_path = random.choice(self.image_files)
        img = self._read_image(img_path)  # HWC float in [0,255
        H, W, C = img.shape
        ps = self.patch_size

        if H < ps or W < ps:
            scale = max(ps / H, ps / W)
            newW = int(round(W * scale))
            newH = int(round(H * scale))
            img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)
            if img.ndim == 2:
                img = img[..., None]
            H, W, C = img.shape

        # random crop
        crop_x = random.randint(0, W - ps)
        crop_y = random.randint(0, H - ps)
        clean = img[crop_y:crop_y + ps, crop_x:crop_x + ps, :]  # HWC

        # augment
        if self.add_augment:
            mode = random.randint(0, 7)
            clean = self._augment(clean, mode).copy()

        # choose ONE task
        task = random.choices(["gauss", "sr", "jpeg"], weights=self.task_probs, k=1)[0]
        noisy = clean.copy()

        if task == "gauss":
            smin, smax = self.sigma_range
            sigma = random.uniform(float(smin), float(smax))  # in [0,55] on 0..255 scale
            noise = np.random.normal(0.0, sigma, size=noisy.shape).astype(np.float32)
            noisy = noisy + noise

        elif task == "sr":
            sf = int(random.choice(self.sr_scales))
            small_w = max(1, int(round(ps / sf)))
            small_h = max(1, int(round(ps / sf)))
            small = cv2.resize(noisy, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
            noisy = cv2.resize(small, (ps, ps), interpolation=cv2.INTER_CUBIC)
            if noisy.ndim == 2:
                noisy = noisy[..., None]

        else:  # jpeg
            qmin, qmax = self.jpeg_quality_range
            q = random.randint(int(qmin), int(qmax))
            tmp = np.clip(noisy, 0, 255).astype(np.uint8)

            if C == 3:
                tmp_bgr = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            else:
                tmp_bgr = tmp[:, :, 0]

            ok, enc = cv2.imencode(".jpg", tmp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError("cv2.imencode failed")

            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR if C == 3 else cv2.IMREAD_GRAYSCALE)
            if C == 3:
                dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                dec = dec.astype(np.float32)[..., None]
            noisy = dec

        # clamp + normalize
        noisy = np.clip(noisy, 0.0, 255.0)
        clean = np.clip(clean, 0.0, 255.0)

        # ensure HWC
        if noisy.ndim == 2:
            noisy = noisy[..., None]
        if clean.ndim == 2:
            clean = clean[..., None]

        inp = np.transpose(noisy, (2, 0, 1)).astype(np.float32) / 255.0
        lab = np.transpose(clean, (2, 0, 1)).astype(np.float32) / 255.0
        return inp, lab


def parse_int_list(s):
    return list(map(int, s.split(","))) if s is not None else None

# =======================================================================================

def get_dataloader(config, images_dir, split="train"):
    sigma_list = parse_int_list(config["gaussian_noise_level"])
    sr_list = parse_int_list(config["downsampling_factor"])
    jpeg_list = parse_int_list(config["jpeg_quality"])

    tp = list(map(float, config["task_probs"].split(",")))
    if len(tp) != 3:
        raise ValueError("task_probs must have 3 values: gauss,sr,jpeg")

    tp_sum = sum(tp)
    task_probs = (tp[0] / tp_sum, tp[1] / tp_sum, tp[2] / tp_sum)

    seed = config["seed"] if split == "train" else config["seed"] + 1

    dataset = Dataset(
        images_dir=images_dir,
        patch_size=config["patch_size"],
        sigma_range=tuple(sigma_list) if sigma_list else (0, 55),
        sr_scales=tuple(sr_list) if sr_list else (2, 3, 4),
        jpeg_quality_range=tuple(jpeg_list) if jpeg_list else (5, 99),
        task_probs=task_probs,
        rgb=False,
        add_augment=True if split == "train" else False,
        steps_per_epoch=config["steps_per_epoch"],
        seed=seed,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["threads"],
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return dataset, dataloader

# =======================================================================================
