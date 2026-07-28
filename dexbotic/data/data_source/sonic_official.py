"""Data source registration for the Unitree G1 SONIC dex_data datasets.

Auto-imported by ``dexbotic/data/data_source/__init__.py`` so that the
registered names become available in ``CONVERSATION_DATA`` (used by
``DexDataset`` via ``dataset_name``).

Produced by ``hardware/unitree_sonic/convert_sonic_to_dexdata.py`` (the -o dir is
itself the dataset folder; no extra sub-folder):
    <out>/jsonl/episode_XXXXX.jsonl              (frames)
    <out>/video/episode_XXXXX_ego_view.mp4       (url in jsonl = bare filename)

So per dataset:
    annotations      = <out>/jsonl        (recursive **/*.jsonl glob)
    data_path_prefix = <out>/video        (joined with each frame's url)
"""
from dexbotic.data.data_source.register import register_dataset

SONIC_DATASET = {
    "dexbotic_pingzi": {
        "data_path_prefix": "./data/dexbotic_pingzi/video",
        "annotations": "./data/dexbotic_pingzi/jsonl",
        "frequency": 1,
    },
    # Pre-extracted image-frame version of dexbotic_pingzi (fast DexDataset path).
    # Produced by hardware/unitree_sonic/extract_frames.py; images_1.type='image'.
    "dexbotic_pingzi_img": {
        "data_path_prefix": "./data/dexbotic_pingzi_img/images",
        "annotations": "./data/dexbotic_pingzi_img/jsonl",
        "frequency": 1,
    },

    "beef_pie_xsh": {
        "data_path_prefix": "./data/beef_pie_xsh/video",
        "annotations": "./data/beef_pie_xsh/jsonl",
        "frequency": 1,
    },
}

# SONIC actions (motion_token + hand joints) are absolute (delta disabled in the
# exp), so delta/periodic masks are not used; keep them empty.
meta_data = {
    "non_delta_mask": None,
    "periodic_mask": None,
    "periodic_range": None,
}

# Registers as ``sonic_bottle_bin``.
register_dataset(SONIC_DATASET, meta_data=meta_data, prefix="sonic")
