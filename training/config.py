import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = {
    "proj": REPO,
    "data_raw": os.path.join(REPO, "data", "raw"),
    "data_proc": os.path.join(REPO, "data", "processed"),
    "logs_dir": os.path.join(REPO, "logs"),
    "models_dir": os.path.join(REPO, "models"),
    "gallery_dir": os.path.join(REPO, "data", "gallery"),

    "cow_cls": 21,

    "h": 256,
    "w": 192,
    "bs": 128,
    "lr": 0.001,
    "lr_min": 1e-6,
    "ep": 100,
    "eval_freq": 5,
    "step": 20,
    "margin": 0.3,
    "weight_t": 1,
    "weight_x": 50,
    "label_smooth": 0.1,
    "num_instances": 4,
    "workers": 8,
    "amp": True,
    "grad_accum": 1,
    "cosine_anneal": True,
    "warmup_epochs": 5,

    "model_name": "osnet_x1_0",

    "aug_n": 3,
    "max_train_per_cow": 50,
    "max_gallery_per_cow": 10,
    "max_query_per_cow": 5,
}


for d in ["data/raw", "data/processed/train", "data/processed/query",
          "data/processed/gallery", "data/gallery", "models", "logs"]:
    os.makedirs(os.path.join(REPO, d), exist_ok=True)
