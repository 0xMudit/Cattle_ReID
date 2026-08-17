# Full Kaggle CLI pipeline: push dataset + notebooks 01-04, poll status, chain outputs.
#
# ONE-TIME SETUP:
#   1. pip install kaggle
#   2. kaggle auth login            # browser OAuth (easiest) -- or put kaggle.json in ~/.kaggle/
#   3. python run_pipeline.py --check
#
# USAGE:
#   python run_pipeline.py --user <kaggle-username> push-videos     # upload Dataset/ videos as a dataset
#   python run_pipeline.py --user <kaggle-username> run             # 02->03->04, chained automatically
#   python run_pipeline.py --user <kaggle-username> run --only 03   # just one notebook
#   python run_pipeline.py --user <kaggle-username> run --resume     # reuse in-flight kernel 02, chain on
#   python run_pipeline.py --user <kaggle-username> status          # status of the last pushed kernels
#   python run_pipeline.py --user <kaggle-username> logs 03         # print last kernel logs
#
# Notes on the Kaggle CLI (2.2.4 / kagglesdk):
#   * `datasets create` requires a `licenses` array in dataset-metadata.json.
#   * `kernels push` checks that slugify(title) == the slug in the metadata `id`,
#     and the server slugs the kernel from the TITLE -- so titles are chosen to
#     slugify to the intended slugs, and the real slug is resolved after push.
#
# Run order and inputs:
#   01 annotate video    in: video dataset              out: annotated mp4 (demo, optional)
#   02 extract crops     in: video dataset              out: crops + meta.json + tracks.json
#   03 train reid        in: crops dataset (from 02)    out: hanwoo_reid.pth + report.json
#   04 vcr eval          in: crops + model (from 02+03) out: vcr comparison
# The chain is: push videos -> run 02 -> pull output -> repack as dataset -> run 03 -> ... -> 04.

import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, glob, tarfile

PY = sys.executable
REPO = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(os.path.dirname(REPO), "Dataset")
WORK = os.path.join(REPO, ".run")          # downloaded kernel outputs land here
CROPS_MIN_BYTES = 50 * 1024 * 1024         # crops dataset must contain the jpgs to be usable

DATASET_VIDEO = "cattle-cctv-videos"
DATASET_WEIGHTS = "cattle-cctv-weights"
DATASET_CROPS = "cattle-cctv-crops"
DATASET_MODEL = "cattle-reid-model"

# title is slugified by Kaggle => keep it resolving to the intended slug
KERNELS = {
    "01": dict(file="01_annotate_video.ipynb", slug="cattle-01-annotate-video",
               title="Cattle 01 annotate video", inputs=[DATASET_VIDEO, DATASET_WEIGHTS]),
    "02": dict(file="02_extract_crops.ipynb", slug="cattle-02-extract-crops",
               title="Cattle 02 extract crops", inputs=[DATASET_VIDEO, DATASET_WEIGHTS]),
    "02b": dict(file="02b_pose_meta.ipynb", slug="cattle-02b-pose-meta",
                title="Cattle 02b pose meta", inputs=[DATASET_CROPS, DATASET_WEIGHTS]),
    "03": dict(file="03_train_hanwoo_reid.ipynb", slug="cattle-03-train-reid",
               title="Cattle 03 train reid", inputs=[DATASET_CROPS]),
    "04": dict(file="04_vcr_eval.ipynb", slug="cattle-04-vcr-eval",
               title="Cattle 04 vcr eval", inputs=[DATASET_CROPS, DATASET_MODEL]),
    "05": dict(file="05_open_set_vcr.ipynb", slug="cattle-05-open-set-vcr",
               title="Cattle 05 open set vcr", inputs=[DATASET_CROPS, DATASET_MODEL]),
    "06": dict(file="06_reid_video_demo.ipynb", slug="cattle-06-reid-video-demo",
               title="Cattle 06 reid video demo", inputs=[DATASET_VIDEO, DATASET_WEIGHTS, DATASET_CROPS, DATASET_MODEL]),
}

def sh(args, **kw):
    r = subprocess.run([PY, "-m", "kaggle"] + args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kw)
    return r

def check(user):
    r = sh(["kernels", "list", "--user", user, "--page-size", "1"])
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        sys.exit("Authentication failed. Run `kaggle auth login` first.")
    print("authenticated as", user)

def dataset_exists(slug):
    owner, name = slug.split("/")
    r = sh(["datasets", "list", "--user", owner, "--page-size", "100"])
    for line in r.stdout.splitlines():
        cols = line.split()
        if cols and cols[0].endswith("/" + name):
            return True
    return False

def create_dataset(folder, slug, title, version=False, min_bytes=1):
    meta = {"id": slug, "title": title, "license": "CC0-1.0",
            "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    if not version and dataset_exists(slug):
        print("  dataset", slug, "already exists -> pushing a new version")
        version = True
    r = sh(["datasets", "version", "-p", folder, "-m", "update"], cwd=folder) if version else sh(["datasets", "create", "-p", folder], cwd=folder)
    if r.returncode != 0 and not version:
        # create failed (e.g. title already taken) -> fall back to versioning
        r = sh(["datasets", "version", "-p", folder, "-m", "update"], cwd=folder)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit("dataset create/version failed for " + slug)
    print(r.stdout.strip() or r.stderr.strip())
    return wait_dataset(slug, min_bytes=min_bytes)

def dataset_files(slug):
    # returns (total_bytes, file_count) from the reliable `datasets list` size column.
    # `datasets files` is paginated at 200 rows and can return nothing while a version
    # is still processing, which made ensure_crops_pushed think crops were missing.
    # The list line is "<ref>  <title with spaces>  <size>  <date>  <downloads> ...", so
    # the size is the token just before the ISO date.
    r = sh(["datasets", "list", "--user", slug.split("/")[0], "--page-size", "100"])
    for line in r.stdout.splitlines():
        if line.startswith(slug + " "):
            toks = line.split()
            for i, t in enumerate(toks[:-1]):
                if re.match(r"^\d{4}-\d{2}-\d{2}", toks[i + 1]):
                    try:
                        return float(t), 1
                    except ValueError:
                        pass
    return 0, 0

def wait_dataset(slug, timeout=5400, poll=20, min_bytes=1):
    # GetDatasetStatus 403s on freshly created datasets; use the list's size column instead.
    name = slug.split("/")[-1]
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = sh(["datasets", "list", "--user", slug.split("/")[0], "--page-size", "100"])
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith(name + " "):
                    cols = line.split()
                    try:
                        sz = float(cols[2])
                        if sz > 0:
                            print("  dataset", slug, "ready (", sz, "bytes )")
                            tb, fc = dataset_files(slug)
                            if tb >= min_bytes:
                                print("  verified", fc, "files,", tb, "bytes on Kaggle")
                                return
                            print(f"  WARNING: only {fc} files / {tb} bytes on Kaggle "
                                  f"(need >= {min_bytes}) -- upload looks incomplete")
                            return False
                    except (ValueError, IndexError):
                        pass
        time.sleep(poll)
    print("  dataset", slug, "never became ready")
    sys.exit(1)

def push_videos(user):
    if not os.path.isdir(VIDEO_DIR):
        sys.exit("no Dataset/ folder at " + VIDEO_DIR)
    vids = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print("uploading", len(vids), "videos")
    tmp = tempfile.mkdtemp(prefix="kaggle_vids_")
    for v in vids:
        shutil.copy2(v, os.path.join(tmp, os.path.basename(v)))
    create_dataset(tmp, user + "/" + DATASET_VIDEO, "Cattle CCTV videos")
    shutil.rmtree(tmp, ignore_errors=True)
    print("dataset pushed:", user + "/" + DATASET_VIDEO)
    push_weights(user)

def push_weights(user):
    wts = [os.path.join(os.path.dirname(REPO), "cattle_osnet", "yolov8n.pt"),
           os.path.join(os.path.dirname(REPO), "cattle_osnet", "models", "cow_pose.pt")]
    tmp = tempfile.mkdtemp(prefix="kaggle_wts_")
    for w in wts:
        if os.path.isfile(w):
            shutil.copy2(w, os.path.join(tmp, os.path.basename(w)))
    create_dataset(tmp, user + "/" + DATASET_WEIGHTS, "Cattle CCTV detector weights")
    shutil.rmtree(tmp, ignore_errors=True)
    print("weights dataset pushed:", user + "/" + DATASET_WEIGHTS)

def build_crops_folder():
    # ds_crops is the folder we push as DATASET_CROPS. The Kaggle CLI (2.2.4) skips real
    # subdirectories but auto-extracts .tar files, nesting the content under a folder
    # named after the tar (crops.tar -> crops/...). So we pack crops with arcname
    # "A1/.." (+ meta.json) -> on Kaggle the dataset root has crops/A1/.., which is the
    # structure the notebooks expect. Uploading 2088 tiny jpgs as one tar is also faster.
    ds = os.path.join(WORK, "ds_crops")
    if not os.path.isfile(os.path.join(ds, "crops.tar")):
        src = os.path.join(WORK, "02_out")
        src_crops = os.path.join(src, "crops")
        if os.path.isdir(src_crops):
            if os.path.isdir(ds): shutil.rmtree(ds)
            os.makedirs(ds)
            tar = os.path.join(ds, "crops.tar")
            with tarfile.open(tar, "w") as tf:
                for name in sorted(os.listdir(src_crops)):
                    tf.add(os.path.join(src_crops, name), arcname=name)
                mp = os.path.join(src, "meta.json")
                if os.path.isfile(mp) and os.path.getsize(mp) > 1024:
                    tf.add(mp, arcname="meta.json")
                else:
                    print("  WARNING: 02_out/meta.json missing or empty -- run make_meta_local.py")
            p = os.path.join(src, "cattle-02-extract-crops.log")
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(ds, "cattle-02-extract-crops.log"))
            print("  ds_crops rebuilt from 02_out (crops.tar with meta.json inside)")
        else:
            sys.exit("no crops found locally (02_out/crops) -- run 02 first")
    return ds

def ensure_crops_pushed(user, min_bytes=CROPS_MIN_BYTES):
    # Last night the crops dataset only contained meta.json (332 B): the jpgs never made
    # it up. Verify what Kaggle actually has and (re)push until the crops are there.
    slug = user + "/" + DATASET_CROPS
    tb, fc = dataset_files(slug)
    print("  crops on Kaggle:", fc, "files /", tb, "bytes (need", min_bytes, ")")
    if tb >= min_bytes:
        return True
    ds = build_crops_folder()
    ok = create_dataset(ds, slug, "Cattle CCTV crops", min_bytes=min_bytes)
    if not ok:
        print("  WARNING: crops upload still not verified -- continuing anyway")
        return False
    return True

def push_crops(user):
    ds = build_crops_folder()
    ok = create_dataset(ds, user + "/" + DATASET_CROPS, "Cattle CCTV crops", min_bytes=CROPS_MIN_BYTES)
    if ok:
        print("crops dataset verified:", user + "/" + DATASET_CROPS)
    else:
        print("crops dataset pushed but upload did not verify")

def resolve_slug(user, title, tries=10, sleep=10):
    # the CLI/server may slug from title or assign a fallback slug; find the real one
    for _ in range(tries):
        r = sh(["kernels", "list", "--user", user, "--page-size", "100"])
        for line in r.stdout.splitlines():
            if title.lower() in line.lower():
                ref = line.split()[0]
                if "/" in ref:
                    return ref
        time.sleep(sleep)
    return None

def push_kernel(user, key):
    cfg = KERNELS[key]
    tmp = tempfile.mkdtemp(prefix="kaggle_nb_")
    shutil.copy2(os.path.join(REPO, cfg["file"]), os.path.join(tmp, cfg["file"]))
    meta = {
        "id": user + "/" + cfg["slug"], "title": cfg["title"],
        "code_file": cfg["file"], "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [user + "/" + d for d in cfg["inputs"]],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(tmp, "Kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    r = sh(["kernels", "push", "-p", tmp])
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); return None
    print(r.stdout.strip() or r.stderr.strip())
    slug = resolve_slug(user, cfg["title"])
    if not slug:
        print("could not resolve slug for", key); return None
    print("resolved", key, "->", slug)
    return slug

def wait(slug, timeout=14400, poll=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = sh(["kernels", "status", slug])
        line = (r.stdout.strip() or r.stderr.strip()).splitlines()[-1]
        print(f"  [{int(time.time()-t0):4d}s] {line}", flush=True)
        low = line.lower()
        if "complete" in low: return "complete"
        if "error" in low or "failed" in low or "cancelled" in low: return "error"
        time.sleep(poll)
    return "timeout"

def run_chain(user, keys, resume=False):
    for key in keys:
        print("=" * 60)
        if key in ("02b", "03", "04"):
            ensure_crops_pushed(user)
        if resume:
            slug = resolve_slug(user, KERNELS[key]["title"])
            if slug:
                print("RESUME", key, "reusing existing kernel", slug)
            else:
                print("PUSH", key, KERNELS[key]["file"])
                slug = push_kernel(user, key)
        else:
            print("PUSH", key, KERNELS[key]["file"])
            slug = push_kernel(user, key)
        if not slug: sys.exit("push failed for " + key)
        st = wait(slug)
        if st != "complete":
            sh(["kernels", "logs", slug])
            sys.exit("kernel " + key + " " + st)
        outdir = os.path.join(WORK, key + "_out")
        os.makedirs(outdir, exist_ok=True)
        if key == "02" and os.path.isfile(os.path.join(outdir, "meta.json")) and os.path.isdir(os.path.join(outdir, "crops")):
            print("  output already pulled locally, skipping download")
        elif key == "03" and os.path.isfile(os.path.join(outdir, "hanwoo_reid.pth")):
            print("  output already pulled locally, skipping download")
        elif any(not f.endswith(".log") for f in os.listdir(outdir)):
            print("  output already pulled locally, skipping download")
        else:
            r = sh(["kernels", "output", slug, "-p", outdir])
        files = [f for f in os.listdir(outdir) if not f.startswith('.')]
        print("  output files:", files)
        if key == "02":
            ds = build_crops_folder()
            create_dataset(ds, user + "/" + DATASET_CROPS, "Cattle CCTV crops", min_bytes=CROPS_MIN_BYTES)
        if key == "02b":
            # 02b regenerated meta.json (pose keypoints were lost last night). Fold it back
            # into the crops folder and re-version the dataset so 03/04 see it.
            mp = os.path.join(outdir, "meta.json")
            if os.path.isfile(mp):
                ds = build_crops_folder()
                shutil.copy2(mp, os.path.join(ds, "meta.json"))
                print("  meta.json folded into ds_crops:", os.path.getsize(mp), "bytes")
                ok = create_dataset(ds, user + "/" + DATASET_CROPS, "Cattle CCTV crops", min_bytes=CROPS_MIN_BYTES)
                if not ok:
                    print("  WARNING: crops re-version did not verify")
            else:
                print("  WARNING: 02b produced no meta.json -- dataset not re-versioned")
        if key == "03":
            ds = os.path.join(WORK, "ds_model")
            if os.path.isdir(ds): shutil.rmtree(ds)
            os.makedirs(ds)
            for f in files:
                p = os.path.join(outdir, f)
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(ds, f))
            create_dataset(ds, user + "/" + DATASET_MODEL, "Cattle ReID model")
    print("=" * 60, "\nALL DONE. Outputs in", WORK)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Kaggle username")
    ap.add_argument("command", choices=["check", "push-videos", "push-weights", "push-crops", "run", "status", "logs"])
    ap.add_argument("--only", help="run only this notebook key, e.g. 03")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing/pending kernel instead of pushing a new one")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if a.command == "check":
        check(a.user)
    elif a.command == "push-videos":
        push_videos(a.user)
    elif a.command == "push-weights":
        push_weights(a.user)
    elif a.command == "push-crops":
        push_crops(a.user)
    elif a.command == "run":
        # 02 extraction is done (output pulled to .run/02_out) and meta.json is
        # regenerated locally by make_meta_local.py. The chain resumes at training:
        # 03 -> 04. (02b was an earlier Kaggle-GPU fixup for the lost meta.json.)
        keys = [a.only] if a.only else ["03", "04"]
        run_chain(a.user, keys, resume=a.resume)
    elif a.command == "status":
        for key, cfg in KERNELS.items():
            r = sh(["kernels", "list", "--user", a.user, "--page-size", "100"])
            ref = None
            for line in r.stdout.splitlines():
                if cfg["title"].lower() in line.lower() and "/" in line:
                    ref = line.split()[0]; break
            if ref:
                s = sh(["kernels", "status", ref])
                print(key, "->", (s.stdout.strip() or s.stderr.strip()).splitlines()[-1])
            else:
                print(key, "-> (not found)")
    elif a.command == "logs":
        cfg = KERNELS[a.only]
        r = sh(["kernels", "list", "--user", a.user, "--page-size", "100"])
        ref = next((ln.split()[0] for ln in r.stdout.splitlines()
                    if cfg["title"].lower() in ln.lower() and "/" in ln), None)
        if ref:
            s = sh(["kernels", "logs", ref])
            print(s.stdout or s.stderr)
        else:
            print("kernel not found for", a.only)

if __name__ == "__main__":
    main()
