import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MoNuSegDataset(CustomDataset):
    """Saliency dataset.
    In segmentation map annotation for Saliency, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False.
    """

    CLASSES = ('background', 'saliency')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(MoNuSegDataset, self).__init__(
            reduce_zero_label=False,
            img_suffix='.png',
            **kwargs)
        assert osp.exists(self.img_dir)