# Upload helper for Kaggle.
# PREREQUISITES (once, on this machine):
#   pip install kaggle
#   put kaggle.json (from https://www.kaggle.com/settings -> Account -> Create New Token)
#   into ~/.kaggle/kaggle.json
#
# Usage:
#   python upload_kaggle.py <kaggle-dataset-name> <path-to-videos-or-folder>
# e.g.
#   python upload_kaggle.py cattle-cctv-videos /path/to/Dataset/video.mp4

import sys, os, subprocess, glob, tempfile, shutil, json

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    name, src = sys.argv[1], sys.argv[2]

    cfg = os.path.expanduser(r"~\.kaggle\kaggle.json")
    if not os.path.exists(cfg):
        print("MISSING", cfg, "- add your Kaggle API token first (see docstring).")
        sys.exit(1)

    tmp = tempfile.mkdtemp(prefix="kaggle_up_")
    try:
        if os.path.isdir(src):
            dst = os.path.join(tmp, name)
            shutil.copytree(src, dst)
        else:
            dst = tmp
            for p in glob.glob(src) or [src]:
                shutil.copy2(p, os.path.join(tmp, os.path.basename(p)))
        # dataset-metadata.json tells Kaggle the dataset name/title
        meta = {"id": name, "title": name}
        with open(os.path.join(tmp, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)
        r = subprocess.run(["kaggle", "datasets", "create", "-p", tmp], capture_output=True, text=True)
        print(r.stdout)
        if r.stderr: print(r.stderr)
        print("Done. Upload videos as input to the notebooks and run them.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
