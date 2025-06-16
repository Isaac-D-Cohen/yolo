

CLASSES = ["NOCA"]
NUM_CLASSES=len(CLASSES)


# Got these anchors from the k_means script
# They represent widths that are scaled to be
# proportions of the image (so between 0 and 1)
ANCHORS = [
    [0.9634, 0.8620, 0.7547],
    [0.6250, 0.4963, 0.3926],
    [0.2919, 0.1747, 0.0756]
]
