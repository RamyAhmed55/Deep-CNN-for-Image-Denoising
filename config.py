# config.py

CONFIGS = {
    "baseline": {
        "arch": "DnCNN-3",
        "gaussian_noise_level": "0,55",
        "downsampling_factor": "2,3,4",
        "jpeg_quality": "5,99",
        "patch_size": 50,
        "batch_size": 128,
        "num_epochs": 50,
        "lr": 1e-3,
        "threads": 4,
        "seed": 123,
        "steps_per_epoch": 500,
        "task_probs": "1,1,1",
        "resume": False,
        "scheduler_milestones": [30, 40, 45],
        "scheduler_gamma": 0.1,
    }
}

# =======================================================================================

def get_config(config_name: str = "baseline"):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    return CONFIGS[config_name].copy()