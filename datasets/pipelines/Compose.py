import collections

from ms_sim.utils.registry import build_from_cfg
from ms_sim.datasets.registry import PIPELINES

@PIPELINES.register_module
class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Empty':
                    continue
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)

            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, res, info):
        for t in self.transforms:
            res, info = t(res, info)
            if res is None:
                return None
        return res, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

# [<ms_sim.datasets.pipelines.loading.LoadPointCloudFromFile object at 0x7fa8a928cc10>,
# <ms_sim.datasets.pipelines.loading.LoadPointCloudAnnotations object at 0x7fa7caefadc0>,
# <ms_sim.datasets.pipelines.preprocess.Preprocess object at 0x7fa7caefac70>,
# <ms_sim.datasets.pipelines.preprocess.Voxelization object at 0x7fa7caefaca0>,
# <ms_sim.datasets.pipelines.formating.Reformat object at 0x7fa7caefad00>]