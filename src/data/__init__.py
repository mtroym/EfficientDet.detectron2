import os

from detectron2.data.datasets.register_coco import register_coco_instances
from .dataset_mapper import DetDatasetMapper

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            {"thing_classes": ["person"]},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_DEEPFASHION2_SPLITS = {
    "deepfashion2_train": ("DeepFashion2/train_real", "DeepFashion2/deepfashion2_train.json"),
    "deepfashion2_val": ("DeepFashion2", "DeepFashion2/deepfashion2_val.json"),
    "deepfashion2_test": ("DeepFashion2/test/image", "DeepFashion2/deepfashion2_test.json"),
    "deepfashion2_toy_val": ("DeepFashion2", "DeepFashion2/deepfashion2_toy_val.json"),
    "deepfashion2_train_55": ("DeepFashion2/train_real", "DeepFashion2/deepfashion2_train_55.json"),
    "deepfashion2_val_55": ("DeepFashion2", "DeepFashion2/deepfashion2_val_55.json"),
}

THING_CLASSES = ["short_sleeved_shirt",
                 "long_sleeved_shirt",
                 "short_sleeved_outwear",
                 "long_sleeved_outwear",
                 "vest",
                 "sling",
                 "shorts",
                 "trousers",
                 "skirt",
                 "short_sleeved_dress",
                 "long_sleeved_dress",
                 "vest_dress",
                 "sling_dress"]

# THING_CLASSES = tuple(THING_CLASSES)

CLOTHS_MAP = [
    ("sst", 25, (1, 4, 6)),
    ("lst", 33, (1, 4, 20)),
    ("sso", 31, (1,)),
    ("lso", 39, (1,)),
    ("vest", 15, (1, 4, 11)),
    ("sling", 15, (1, 4, 11)),
    ("shorts", 10, (2, 7,)),
    ("trousers", 14, (2, 9,)),
    ("skirt", 8, (2, 6,)),
    ("ssd", 29, (1, 4, 18)),
    ("lsd", 37, (1, 4, 22)),
    ("vd", 19, (1, 4, 13)),
    ("sd", 19, (1, 4, 13))
]

KEYPOINT_NAMES = []
for entry in CLOTHS_MAP:
    KEYPOINT_NAMES += ["{}_{}".format(entry[0], id) for id in range(1, entry[1] + 1)]
KEYPOINT_NAMES = tuple(KEYPOINT_NAMES)


def build_flip_pair(name, id1, id2):
    return ("{}_{}".format(name, id1),
            "{}_{}".format(name, id2))


def build_filp_pairs(name, ids1, ids2):
    assert len(ids1) == len(ids2)
    length = len(ids1)
    return [build_flip_pair(name, ids1[i], ids2[i]) for i in range(length)]


KEYPOINT_FLIP_MAP = [
    # sst------------------
    *build_filp_pairs("sst",
                      [2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      [6, 5, 25, 24, 23, 22, 21, 20, 19, 18, 17]),
    *build_filp_pairs("lst",
                      [2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      [6, 5, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21]),
    *build_filp_pairs("sso",
                      [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 31],
                      [26, 5, 6, 25, 24, 23, 22, 21, 20, 19, 18, 17, 29, 27, 28]),
    *build_filp_pairs("lso",
                      [2, 3, 4, *[i for i in range(7, 21)], 39, 38],
                      [6, 5, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 37, 36, 35]),
    *build_filp_pairs("vest",
                      [2, 3, 7, 8, 9, 10],
                      [6, 5, 15, 14, 13, 12]),
    *build_filp_pairs("sling",
                      [2, 3, 7, 8, 9, 10],
                      [6, 5, 15, 14, 13, 12]),
    *build_filp_pairs("shorts",
                      [1, 4, 5, 6],
                      [3, 10, 9, 8]),
    *build_filp_pairs("trousers",
                      [1, 4, 5, 6, 7, 8],
                      [3, 14, 13, 12, 11, 10]),
    *build_filp_pairs("skirt",
                      [1, 4, 5],
                      [3, 8, 7]),
    *build_filp_pairs("ssd",
                      [2, 3, *[i for i in range(7, 18)]],
                      [6, 5, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19]),
    *build_filp_pairs("lsd",
                      [2, 3, *[i for i in range(7, 22)]],
                      [6, 5, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28,
                       27, 26, 25, 24, 23]),
    *build_filp_pairs("vd",
                      [2, 3, *[i for i in range(7, 13)]],
                      [6, 5, 19, 18, 17, 16, 15, 14]),
    *build_filp_pairs("sd",
                      [2, 3, *[i for i in range(7, 13)]],
                      [6, 5, 19, 18, 17, 16, 15, 14]),
]
KEYPOINT_FLIP_MAP = tuple(KEYPOINT_FLIP_MAP)
CONNECTION_RULES = ()

df2_metadata = {
    # "thing_classes": THING_CLASSES,
    "keypoint_names": KEYPOINT_NAMES,
    "keypoint_flip_map": KEYPOINT_FLIP_MAP,
    "keypoint_connection_rules": CONNECTION_RULES,
}

KEYPOINT_FLIP_MAP_55 = [
    *build_filp_pairs("all",
                      [2, 3, 41, *[i for i in range(7, 20, 1)], 41, 40, 39, 38, 42, 43, 44, 45, 46],
                      [6, 5, 34, *[i for i in range(33, 20, -1)], 34, 35, 36, 37, 52, 51, 50, 49, 48]),
]

KEYPOINT_FLIP_MAP_55 = tuple(KEYPOINT_FLIP_MAP_55)

df2_55_metadata = {
    "keypoint_names": tuple([f"all_{i}" for i in range(1, 56)]),
    "keypoint_flip_map": KEYPOINT_FLIP_MAP_55,
    "keypoint_connection_rules": CONNECTION_RULES,
}


def register_all_df2(root="datasets"):
    for key, (image_root, json_file) in _DEEPFASHION2_SPLITS.items():
        register_coco_instances(
            key,
            metadata=df2_55_metadata if "55" in json_file else df2_metadata,
            json_file=os.path.join(root, json_file) if "://" not in json_file else json_file,
            image_root=os.path.join(root, image_root),
        )


if __name__ == '__main__':
    print(KEYPOINT_NAMES)
    print(KEYPOINT_FLIP_MAP)
    KEYPOINT_FLIP_MAP_INDEX = [
        (KEYPOINT_NAMES.index(name_pair[0]), KEYPOINT_NAMES.index(name_pair[1])) for name_pair in KEYPOINT_FLIP_MAP
    ]
    print(KEYPOINT_FLIP_MAP_INDEX)
else:
    register_all_coco()
    # register_all_df2()
