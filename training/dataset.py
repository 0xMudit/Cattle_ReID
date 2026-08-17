#!/usr/bin/env python3
"""Custom torchreid dataset for cattle images."""
import os
import glob
import torchreid
from torchreid.data.datasets import ImageDataset

from .config import CFG


class CattleDS(ImageDataset):
    """Custom dataset reading from separate train/query/gallery directories."""

    def __init__(self, root="", **kw):
        proc = CFG["data_proc"]
        train_dir = os.path.join(proc, "train")
        query_dir = os.path.join(proc, "query")
        gallery_dir = os.path.join(proc, "gallery")

        super().__init__(
            self._pd(train_dir, False),
            self._pd(query_dir, True),
            self._pd(gallery_dir, False),
            **kw,
        )

    def _pd(self, d, is_query):
        data = []
        if not os.path.isdir(d):
            return data
        for p in glob.glob(os.path.join(d, "*.jpg")):
            try:
                nm = os.path.basename(p).split("_")
                pid = int(nm[1][1:])
                camid = int(nm[0][1:])
                if is_query:
                    camid += 10
                data.append((p, pid, camid))
            except (IndexError, ValueError):
                pass
        return data


def register_cattle_dataset():
    """Register CattleDS with torchreid and return the dataset name."""
    import random
    import string
    dn = "cattle_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    torchreid.data.register_image_dataset(dn, CattleDS)
    return dn
