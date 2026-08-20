# Core ML frameworks
!pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Person ReID library (provides OSNet and training utilities)
!pip install -q git+https://github.com/KaiyangZhou/deep-person-reid.git

# Object detection, augmentation, export
!pip install -q albumentations onnx onnxruntime ultralytics matplotlib tqdm

# Contrastive learning (optional, for self-supervised pre-training)
!pip install -q pytorch_metric_learning

print('✅ All dependencies installed')

import os

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJ = '/kaggle/working/cattle_reid'  # Kaggle working directory (persists across sessions)
# For Google Colab, use: PROJ = '/content/cattle_reid'

# Create directory structure
DIRS = [
    'data/raw',              # Downloaded CID dataset
    'data/processed/train',  # Training images (70% of cows)
    'data/processed/gallery',# Gallery images (15% of cows)
    'data/processed/query',  # Query/test images (15% of cows)
    'data/gallery',          # Registered gallery embeddings
    'models',                # Exported ONNX models
    'logs',                  # Training logs and checkpoints
]
for d in DIRS:
    os.makedirs(f'{PROJ}/{d}', exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
CFG = {
    # Backbone selection: 'osnet_x1_0' | 'resnet18' | 'efficientnet_b0' | 'mobilenet_v3_small'
    'backbone': 'osnet_x1_0',

    # Input dimensions (height, width)
    'h': 256,
    'w': 192,

    # Training hyperparameters
    'bs': 32,           # Batch size (reduce to 16 if OOM)
    'lr': 0.003,        # Learning rate
    'ep': 30,           # Number of epochs
    'eval_freq': 5,     # Evaluate every N epochs
    'step': 10,         # LR scheduler step (decay at this epoch)

    # Triplet loss parameters
    'margin': 0.3,      # Minimum distance between different cows
    'weight_t': 1,      # Triplet loss weight
    'weight_x': 50,     # Softmax loss weight

    # Matching
    'threshold': 0.6,   # L2 distance threshold for identification

    # Data split
    'train_pct': 0.70,  # 70% of cow IDs for training
    'gallery_pct': 0.15,# 15% for gallery (enrollment)
    # Remaining 15% for query (testing)

    # Data processing
    'max_train_imgs': 50,   # Max images per cow for training
    'max_gallery_imgs': 10, # Max images per cow for gallery
    'max_query_imgs': 5,    # Max images per cow for query
    'augment_n': 3,         # Number of augmented copies per training image

    # YOLO detection
    'cow_class': 19,        # COCO class ID for cow (0-based)
    'det_conf': 0.25,       # Minimum detection confidence
}

print(f'Project: {PROJ}')
print(f'Backbone: {CFG["backbone"]}')
print(f'Input: {CFG["h"]}x{CFG["w"]}')
print(f'Training: {CFG["ep"]} epochs, batch_size={CFG["bs"]}')

# CID dataset URLs
CID_URLS = {
    'images': 'https://cid-21.s3.amazonaws.com/images.tar.gz',
    'yt_images': 'https://cid-21.s3.amazonaws.com/yt_images.tar.gz',
    'metadata': 'https://cid-21.s3.amazonaws.com/dataset.csv',
}

# Download only if not already present
for name, url in CID_URLS.items():
    dest = f'{PROJ}/data/raw/{name}.tar.gz' if name != 'metadata' else f'{PROJ}/data/raw/metadata.csv'
    if not os.path.exists(dest):
        print(f'Downloading {name}...')
        !curl -L --progress-bar -o {dest} {url}
    else:
        print(f'{name} already exists, skipping download')

print('✅ Download complete')

import tarfile

# Extract tar.gz archives
for tar_name in ['images.tar.gz', 'yt_images.tar.gz']:
    tar_path = f'{PROJ}/data/raw/{tar_name}'
    if os.path.exists(tar_path):
        print(f'Extracting {tar_name}...')
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall(f'{PROJ}/data/raw/')
        print(f'  Extracted to {PROJ}/data/raw/')

# Verify extraction
RAW_DIR = f'{PROJ}/data/raw/images'
if os.path.exists(RAW_DIR):
    cow_dirs = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    print(f'✅ Found {len(cow_dirs)} cow identity directories')
else:
    print(f'⚠️  Images directory not found at {RAW_DIR}')

import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load YOLOv8n (nano — smallest and fastest)
# First run will auto-download yolov8n.pt (~6MB)
yolo = YOLO('yolov8n.pt').to(device)

# Verify cow detection works
COW_CLS = CFG['cow_class']  # Should be 19
print(f'COW_CLS = {COW_CLS} (should be 19 for cow)')
print('✅ YOLOv8 loaded')

import albumentations as A

class Prep:
    """
    Preprocessing pipeline for cattle images.
    
    Handles detection, cropping, letterbox resize, and augmentation.
    
    Args:
        model: YOLO model for cow detection
        h: Target image height
        w: Target image width
        letterbox: If True, preserve aspect ratio with padding
    """
    def __init__(self, model, h=256, w=192, letterbox=True):
        self.m = model
        self.h = h
        self.w = w
        self.letterbox = letterbox

        # Augmentation pipeline (training only)
        self.aug = A.Compose([
            A.GaussNoise(var_limit=(10, 80), p=0.33),
            A.Blur(blur_limit=3, p=0.33),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.2),
                contrast_limit=(-0.3, 0.2), p=0.3
            ),
            A.CLAHE(clip_limit=4, p=0.3),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                hue=0.1, saturation=0.3, p=0.33
            ),
            A.CoarseDropout(
                max_holes=8, max_height=16,
                max_width=16, p=0.33
            ),
            A.HorizontalFlip(p=0.5),
        ])

    def detect(self, img):
        """
        Run YOLO detection on an image.
        
        Returns:
            Nx4 array of bounding boxes [x1, y1, x2, y2] for cow detections,
            or empty array if no cows found.
        """
        r = self.m(img, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=int)

        # Filter to cow class only
        mask = boxes.cls.cpu().numpy() == COW_CLS
        if not mask.any():
            return np.empty((0, 4), dtype=int)

        xyxy = boxes.xyxy.cpu().numpy()[mask].astype(int)
        return xyxy

    def crop(self, img, bb):
        """
        Crop and resize with aspect-ratio-preserving letterbox.
        
        Letterbox resize scales the image to fit within (h, w)
        while keeping the original aspect ratio. Empty space is
        filled with black pixels (padding).
        """
        x1, y1, x2, y2 = bb
        crop = img[y1:y2, x1:x2]

        if not self.letterbox:
            return cv2.resize(crop, (self.w, self.h))

        # Letterbox: preserve aspect ratio
        h0, w0 = crop.shape[:2]
        if h0 == 0 or w0 == 0:
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)

        scale = min(self.h / h0, self.w / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(crop, (new_w, new_h))

        # Center-pad to target size
        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        pad_w = (self.w - new_w) // 2
        pad_h = (self.h - new_h) // 2
        out[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return out

    def aug_img(self, img, n=3):
        """
        Create n augmented versions of an image.
        
        Returns:
            List of [original] + n augmented images
        """
        imgs = [img]
        for _ in range(n):
            imgs.append(self.aug(image=img)['image'])
        return imgs


# Initialize preprocessing pipeline
prep = Prep(yolo, h=CFG['h'], w=CFG['w'])
print('✅ Preprocessing pipeline ready')

from glob import glob
from pathlib import Path
from tqdm import tqdm

RAW = f'{PROJ}/data/raw'
PROC = f'{PROJ}/data/processed'

# ===========================================================================
# Step 1: Collect all images and group by cow ID
# ===========================================================================
all_imgs = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    all_imgs.extend(glob(f'{RAW}/**/{ext}', recursive=True))

# Group images by cow ID (extracted from directory names)
cows = {}  # {cow_id: [image_paths]}
for p in all_imgs:
    for part in Path(p).parts:
        if part.isdigit():
            cows.setdefault(int(part), []).append(p)
            break

print(f'Found {len(cows)} cows with {sum(len(v) for v in cows.values())} images total')

# ===========================================================================
# Step 2: Split cow IDs into train / gallery / query
# ===========================================================================
cow_ids = sorted(cows.keys())
n = len(cow_ids)
n_train = int(n * CFG['train_pct'])
n_gallery = int(n * CFG['gallery_pct'])

train_ids = set(cow_ids[:n_train])
gallery_ids = set(cow_ids[n_train:n_train + n_gallery])
query_ids = set(cow_ids[n_train + n_gallery:])

print(f'Split: {len(train_ids)} train, {len(gallery_ids)} gallery, {len(query_ids)} query identities')

# ===========================================================================
# Step 3: Process images (detect, crop, augment, save)
# ===========================================================================
counts = {'train': 0, 'gallery': 0, 'query': 0}
skipped = 0

for cid, paths in tqdm(cows.items(), desc='Processing cows'):
    # Determine which split this cow belongs to
    if cid in train_ids:
        subset = paths[:CFG['max_train_imgs']]
        dest = f'{PROC}/train'
        augment = True
    elif cid in gallery_ids:
        subset = paths[:CFG['max_gallery_imgs']]
        dest = f'{PROC}/gallery'
        augment = False
    elif cid in query_ids:
        subset = paths[:CFG['max_query_imgs']]
        dest = f'{PROC}/query'
        augment = False
    else:
        continue

    for p in subset:
        im = cv2.imread(p)
        if im is None:
            continue

        # Detect cows
        bbs = prep.detect(im)
        if len(bbs) == 0:
            skipped += 1
            continue  # YOLO missed — skip this image

        # Crop the first detected cow
        cropped = prep.crop(im, bbs[0])

        # Apply augmentation if training split
        images = prep.aug_img(cropped, CFG['augment_n']) if augment else [cropped]

        # Save processed images
        split_name = dest.split('/')[-1]
        for img_aug in images:
            counts[split_name] += 1
            name = f'c0_p{cid}_{counts[split_name]}.jpg'
            cv2.imwrite(f'{dest}/{name}', img_aug)

print(f'\n✅ Processing complete:')
print(f'  Train: {counts["train"]} images')
print(f'  Gallery: {counts["gallery"]} images')
print(f'  Query: {counts["query"]} images')
print(f'  Skipped (no cow detected): {skipped}')

import os
import glob
import random
import string
import torch
import torchreid
from torchreid.data.datasets import ImageDataset

TRAIN_DIR = f'{PROC}/train'
QUERY_DIR = f'{PROC}/query'
GALLERY_DIR = f'{PROC}/gallery'


class CattleDS(ImageDataset):
    """
    Custom dataset for cattle re-identification.
    
    Reads from separate train/query/gallery directories.
    Query images get a fake camera ID shift (+10) because
    the CID dataset lacks real camera metadata.
    
    Filenames follow format: c<camid>_p<pid>_<count>.jpg
    """
    def __init__(self, root='', **kw):
        super().__init__(
            self._parse_dir(TRAIN_DIR, is_query=False),
            self._parse_dir(QUERY_DIR, is_query=True),
            self._parse_dir(GALLERY_DIR, is_query=False),
            **kw
        )

    def _parse_dir(self, dir_path, is_query):
        """Parse a directory of cow images into (path, pid, camid) tuples."""
        data = []
        if not os.path.isdir(dir_path):
            return data
        for p in glob.glob(os.path.join(dir_path, '*.jpg')):
            try:
                name_parts = os.path.basename(p).split('_')
                pid = int(name_parts[1][1:])     # p<COW_ID> -> COW_ID
                camid = int(name_parts[0][1:])    # c<CAM_ID> -> CAM_ID
                if is_query:
                    camid += 10  # Shift camera ID for query images
                data.append((p, pid, camid))
            except (IndexError, ValueError):
                pass  # Skip files with unexpected naming
        return data


# Print dataset stats
print(f'Train dir: {len(os.listdir(TRAIN_DIR))} files')
print(f'Gallery dir: {len(os.listdir(GALLERY_DIR))} files')
print(f'Query dir: {len(os.listdir(QUERY_DIR))} files')

from torchreid.engine import ImageTripletEngine

# Register dataset with torchreid (requires random name to avoid collisions)
dataset_name = 'cattle_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
torchreid.data.register_image_dataset(dataset_name, CattleDS)

# Data manager: handles loading, transforms, batching
data_manager = torchreid.data.ImageDataManager(
    sources=dataset_name,
    height=CFG['h'],
    width=CFG['w'],
    batch_size_train=CFG['bs'],
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
)

# Build model
model = torchreid.models.build_model(
    name=CFG['backbone'],
    num_classes=data_manager.num_train_pids,
    loss='triplet',
    pretrained=True,
).to(device)

# Optimizer and LR scheduler
optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=CFG['lr'])
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=CFG['step']
)

# Training engine (combines triplet + softmax loss)
engine = ImageTripletEngine(
    data_manager, model,
    optimizer=optimizer,
    scheduler=scheduler,
    margin=CFG['margin'],
    weight_t=CFG['weight_t'],
    weight_x=CFG['weight_x'],
)

n_params = sum(p.numel() for p in model.parameters())
print(f'\n✅ Model ready on {device}')
print(f'  Architecture: {CFG["backbone"]}')
print(f'  Parameters: {n_params:,}')
print(f'  Train identities: {data_manager.num_train_pids}')
print(f'  Query images: {len(data_manager.query_dataset)}')
print(f'  Gallery images: {len(data_manager.gallery_dataset)}')

print(f'Starting training for {CFG["ep"]} epochs...')
print(f'Evaluating every {CFG["eval_freq"]} epochs')
print(f'Logs saved to: {PROJ}/logs/{CFG["backbone"]}/')
print('=' * 60)

engine.run(
    save_dir=f'{PROJ}/logs/{CFG["backbone"]}',
    max_epoch=CFG['ep'],
    eval_freq=CFG['eval_freq'],
    print_freq=50,
)

print('\n✅ Training complete!')

import pickle
from torchreid.utils import FeatureExtractor


class Registry:
    """
    Gallery of known cows with their embedding signatures.
    
    Stores embeddings for each registered cow and computes
    the mean embedding for matching.
    """
    def __init__(self, name='osnet_x1_0', model_path=None):
        self.gal = {}  # {cow_name: {'embs': array, 'mean': array, 'n': int}}
        self.gal_file = f'{PROJ}/data/gallery/gal.pkl'

        # Feature extractor for generating embeddings
        self.ext = FeatureExtractor(
            model_name=name,
            model_path=model_path,
            device=device,
            image_size=(CFG['h'], CFG['w']),
            verbose=False,
        )

        # Load existing gallery if available
        if os.path.exists(self.gal_file):
            with open(self.gal_file, 'rb') as f:
                self.gal = pickle.load(f)
            print(f'Loaded existing gallery: {len(self.gal)} cows')

    def register(self, name, imgs):
        """
        Register a cow with multiple images.
        
        Args:
            name: Cow identifier (e.g., 'Cow_001')
            imgs: List of image paths or numpy arrays
        """
        embs = []
        for im in imgs:
            if isinstance(im, str):
                im = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
            emb = self.ext([im]).cpu().detach().numpy().flatten()
            embs.append(emb)

        self.gal[name] = {
            'embs': np.array(embs),
            'mean': np.mean(embs, axis=0),
            'n': len(imgs),
        }

        # Save gallery to disk
        with open(self.gal_file, 'wb') as f:
            pickle.dump(self.gal, f)
        print(f'Registered {name}: {len(imgs)} images')

    def names(self):
        """Return list of all registered cow names."""
        return list(self.gal.keys())

    def remove(self, name):
        """Remove a cow from the gallery."""
        if name in self.gal:
            del self.gal[name]
            with open(self.gal_file, 'wb') as f:
                pickle.dump(self.gal, f)


# Initialize registry
registry = Registry(name=CFG['backbone'])
print('✅ Gallery registry ready')

# Register gallery cows from processed gallery directory
gallery_imgs = glob.glob(f'{PROC}/gallery/*.jpg')
gallery_cows = {}  # {cow_id: [image_paths]}

for p in gallery_imgs:
    parts = os.path.basename(p).split('_')
    try:
        cid = int(parts[1][1:])  # p<COW_ID> -> COW_ID
        gallery_cows.setdefault(cid, []).append(p)
    except (IndexError, ValueError):
        pass

# Register each gallery cow
for cid, paths in gallery_cows.items():
    registry.register(f'Cow_{cid:03d}', paths[:CFG['max_gallery_imgs']])

print(f'\n✅ Registered {len(registry.names())} cows in gallery')
print(f'Cows: {registry.names()[:10]}...' if len(registry.names()) > 10 else f'Cows: {registry.names()}')
print(f'\nNote: Query cows (not in gallery) will appear as "Unknown" — this is correct!')

class Recognizer:
    """
    End-to-end cattle recognition pipeline.
    
    Detects cows in an image, extracts their embedding,
    and matches against the gallery of known cows.
    
    Args:
        registry: Gallery of known cows
        yolo: YOLO model for detection
        threshold: Maximum L2 distance for a match
    """
    def __init__(self, registry, yolo, threshold=0.6):
        self.reg = registry
        self.yolo = yolo
        self.thr = threshold

    def l2_distance(self, a, b):
        """Compute L2 (Euclidean) distance between two embeddings."""
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def run(self, img):
        """
        Run recognition on an image.
        
        Args:
            img: BGR numpy array (from cv2.imread)
        
        Returns:
            List of dicts with 'id', 'conf', 'dist', 'bbox', 'det_conf'
        """
        results = []

        # Step 1: Detect cows
        r = self.yolo(img, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return results

        # Filter to cow class
        mask = boxes.cls.cpu().numpy() == COW_CLS
        if not mask.any():
            return results

        xyxy = boxes.xyxy.cpu().numpy()[mask].astype(int)
        confs = boxes.conf.cpu().numpy()[mask]

        if not self.reg.gal:
            return results

        # Step 2: For each detected cow
        for bb, det_conf in zip(xyxy, confs):
            x1, y1, x2, y2 = bb

            # Crop and preprocess
            crop = cv2.cvtColor(
                cv2.resize(img[y1:y2, x1:x2], (CFG['w'], CFG['h'])),
                cv2.COLOR_BGR2RGB
            )

            # Step 3: Extract embedding
            emb = self.reg.ext([crop]).cpu().detach().numpy().flatten()

            # Step 4: Match against gallery
            best_id = None
            best_dist = float('inf')
            for cow_name, cow_data in self.reg.gal.items():
                dist = self.l2_distance(emb, cow_data['mean'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = cow_name

            # Step 5: Determine if known or unknown
            is_known = best_dist < self.thr
            cow_id = best_id if is_known else 'Unknown'
            confidence = 1.0 - (best_dist / self.thr) if is_known else 0.0

            results.append({
                'id': cow_id,
                'conf': confidence,
                'dist': best_dist,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'det_conf': float(det_conf),
            })

        return results

    def draw(self, img, results):
        """
        Draw bounding boxes and labels on the image.
        
        Green = identified cow
        Red = unknown cow
        """
        vis = img.copy()
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            color = (0, 255, 0) if r['id'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            label = f"{r['id']} {r['conf']:.2f}"
            cv2.putText(
                vis, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        return vis


# Initialize recognizer
recognizer = Recognizer(registry, yolo, threshold=CFG['threshold'])
print('✅ Recognizer ready')

import matplotlib.pyplot as plt

# Get query images
query_imgs = glob.glob(f'{PROC}/query/*.jpg')

if query_imgs:
    # Test on first 5 query images
    n_test = min(5, len(query_imgs))
    fig, axes = plt.subplots(1, n_test, figsize=(5 * n_test, 5))
    if n_test == 1:
        axes = [axes]

    for i, img_path in enumerate(query_imgs[:n_test]):
        im = cv2.imread(img_path)
        results = recognizer.run(im)
        vis = recognizer.draw(im, results)

        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f'{os.path.basename(img_path)}', fontsize=10)

        # Print results
        for r in results:
            status = '✅' if r['id'] != 'Unknown' else '❓'
            print(f'  {status} {r["id"]}: conf={r["conf"]:.2f}  dist={r["dist"]:.3f}  det_conf={r["det_conf"]:.2f}')

    plt.tight_layout()
    plt.show()

    print(f'\n✅ Tested on {n_test} query images')
else:
    print('⚠️  No query images found. Run the processing cell first.')

# Upload an image (Kaggle/Colab)
# For Kaggle: Use the Data panel on the right to upload
# For Colab: Use files.upload()

CUSTOM_IMAGE_PATH = None  # Set this to your image path

# Example: if you uploaded 'cctv_frame.jpg'
# CUSTOM_IMAGE_PATH = '/kaggle/input/cctv-frame.jpg'

if CUSTOM_IMAGE_PATH and os.path.exists(CUSTOM_IMAGE_PATH):
    im = cv2.imread(CUSTOM_IMAGE_PATH)
    results = recognizer.run(im)
    vis = recognizer.draw(im, results)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Recognition Results: {os.path.basename(CUSTOM_IMAGE_PATH)}')
    plt.show()

    # Print detailed results
    print(f'\nDetected {len(results)} cow(s):')
    for r in results:
        status = 'KNOWN' if r['id'] != 'Unknown' else 'UNKNOWN'
        print(f'  [{status}] ID: {r["id"]}')
        print(f'    Confidence: {r["conf"]:.2%}')
        print(f'    Distance: {r["dist"]:.4f}')
        print(f'    Detection confidence: {r["det_conf"]:.2%}')
        print()
else:
    print('Set CUSTOM_IMAGE_PATH above to test on your own image.')
    print('Example: CUSTOM_IMAGE_PATH = \'/kaggle/input/my-photo.jpg\'')

import onnx

# Find the best checkpoint
checkpoints = glob.glob(f'{PROJ}/logs/{CFG["backbone"]}/model/model.pth.tar-*')

if checkpoints:
    # Load the latest checkpoint
    best_ckpt = max(checkpoints, key=os.path.getctime)
    print(f'Loading checkpoint: {best_ckpt}')

    # Rebuild model and load weights
    export_model = torchreid.models.build_model(
        name=CFG['backbone'],
        num_classes=data_manager.num_train_pids,
    ).to(device)
    torchreid.utils.load_pretrained_weights(export_model, best_ckpt)
    export_model.eval()

    # Export to ONNX
    onnx_path = f'{PROJ}/models/cattle_reid.onnx'
    dummy_input = torch.randn(1, 3, CFG['h'], CFG['w']).to(device)

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    # Verify the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Print model info
    model_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'\n✅ ONNX model exported successfully!')
    print(f'  Path: {onnx_path}')
    print(f'  Size: {model_size:.1f} MB')
    print(f'  Input shape: {dummy_input.shape}')
    print(f'  Output shape: {export_model(dummy_input).shape}')
else:
    print('⚠️  No checkpoint found. Run training first.')

# K-Fold evaluation (run after training)
# This gives more reliable metrics than a single split

from collections import defaultdict


class KFoldEvaluator:
    """
    K-Fold cross-validation evaluator.
    Splits cow IDs into K folds, trains on K-1, tests on 1.
    """
    def __init__(self, k_folds=5, seed=12345):
        self.k = k_folds
        self.seed = seed

    def split_cows(self, cow_ids):
        """Split cow IDs into K folds."""
        rng = np.random.RandomState(self.seed)
        shuffled = cow_ids.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, self.k)

        splits = []
        for k in range(self.k):
            test_ids = set(folds[k].tolist())
            train_ids = set()
            for j in range(self.k):
                if j != k:
                    train_ids.update(folds[j].tolist())
            splits.append((train_ids, test_ids))
        return splits

    def evaluate(self, cow_ids, embeddings_dict, threshold=0.6):
        """Run K-fold evaluation."""
        splits = self.split_cows(cow_ids)
        results = []

        for fold, (train_ids, test_ids) in enumerate(splits):
            # Build gallery from train cows
            gallery = {}
            for cid in train_ids:
                if cid in embeddings_dict and len(embeddings_dict[cid]) > 0:
                    embs = np.array(embeddings_dict[cid])
                    gallery[cid] = np.mean(embs, axis=0)

            if not gallery:
                continue

            # Test on test cows
            correct = 0
            total = 0
            for cid in test_ids:
                if cid not in embeddings_dict:
                    continue
                for emb in embeddings_dict[cid]:
                    best_dist = float('inf')
                    best_match = None
                    for gid, g_emb in gallery.items():
                        dist = np.sqrt(np.mean((emb - g_emb) ** 2))
                        if dist < best_dist:
                            best_dist = dist
                            best_match = gid
                    if best_match == cid and best_dist < threshold:
                        correct += 1
                    total += 1

            acc = (correct / max(total, 1)) * 100
            results.append(acc)
            print(f'  Fold {fold+1}: {acc:.1f}% accuracy')

        avg = np.mean(results)
        std = np.std(results)
        print(f'\n  Average: {avg:.1f}% +/- {std:.1f}%')
        return avg, std


# Uncomment to run K-Fold evaluation:
# kfold = KFoldEvaluator(k_folds=5)
# kfold.evaluate(cow_ids, embeddings_dict)

print('=' * 60)
print('CATTLE RE-IDENTIFICATION — PIPELINE SUMMARY')
print('=' * 60)

print(f'\n📊 Dataset:')
print(f'  Total cows: {len(cows)}')
print(f'  Train identities: {len(train_ids)}')
print(f'  Gallery identities: {len(gallery_ids)}')
print(f'  Query identities: {len(query_ids)}')

print(f'\n🔧 Model:')
print(f'  Backbone: {CFG["backbone"]}')
print(f'  Parameters: {n_params:,}')
print(f'  Input size: {CFG["h"]}x{CFG["w"]}')
print(f'  Device: {device}')

print(f'\n⚙️ Training:')
print(f'  Epochs: {CFG["ep"]}')
print(f'  Batch size: {CFG["bs"]}')
print(f'  Learning rate: {CFG["lr"]}')

print(f'\n🎯 Matching:')
print(f'  Gallery size: {len(registry.names())} cows')
print(f'  Threshold: {CFG["threshold"]}')

print(f'\n📁 Outputs:')
print(f'  Checkpoints: {PROJ}/logs/{CFG["backbone"]}/')
print(f'  ONNX model: {PROJ}/models/cattle_reid.onnx')
print(f'  Gallery: {PROJ}/data/gallery/gal.pkl')

print(f'\n' + '=' * 60)
print('✅ Pipeline complete!')
print('=' * 60)
