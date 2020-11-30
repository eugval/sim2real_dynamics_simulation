'''
Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite

'''

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.models.objects.generated_objects import _get_size , _get_randomized_range, DEFAULT_DENSITY_RANGE
import numpy as np

class FullyFrictionalBoxObject(MujocoGeneratedObject):
    """ Generates a mujoco object than can have different values for the sliding and torsional frictions"""
    def __init__(
            self,
            size=None,
            size_max=None,
            size_min=None,
            density=None,
            density_range=None,
            friction=None,
            rgba="random",
    ):


        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07, 0.07],
                         [0.03, 0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)

        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
        )

    def get_collision_attrib_template(self):
        super_template = super().get_collision_attrib_template()
        super_template['condim'] = '4'
        return super_template

    def sanity_check(self):
        assert len(self.size) == 3, "box size should have length 3"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="box")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="box")