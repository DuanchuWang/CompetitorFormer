from .scannetv2 import ScanNetDataset
from .scannet200_constants import NYU_ID, CLASSES

class ScanNet200Dataset(ScanNetDataset):

    CLASSES = CLASSES
    NYU_ID = NYU_ID[2:]