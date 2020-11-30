'''
Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite

'''

from robosuite.models.tasks.placement_sampler import  ObjectPositionSampler, UniformRandomSampler
import collections
import numpy as np


def base_sample_new(self, **kwargs):
    """
    Args:
        object_index: index of the current object being sampled
    Returns:
        xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
        xquat((float * 4) * n_obj): quaternion of the objects
    """
    raise NotImplementedError

def sample_quat_new(self):
    if self.z_rotation is None or self.z_rotation is True:
        rot_angle = np.random.uniform(high=2 * np.pi, low=0)
    elif isinstance(self.z_rotation, collections.Iterable):
        rot_angle = np.random.uniform(
            high=max(self.z_rotation), low=min(self.z_rotation)
        )
    else:
        rot_angle = self.z_rotation

    return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]

UniformRandomSampler.sample_quat = sample_quat_new
ObjectPositionSampler.sample = base_sample_new



class UniformSelectiveSampler(ObjectPositionSampler):
    """Places all objects within the table uniformly random."""

    def __init__(
            self,
            x_range=None,
            y_range=None,
            ensure_object_boundary_in_range=True,
            z_rotation=False,
            np_random=None
    ):
        """
        Args:
            x_range(float * 2): override the x_range used to uniformly place objects
                    if None, default to x-range of table
            y_range(float * 2): override the y_range used to uniformly place objects
                    if None default to y-range of table
            x_range and y_range are both with respect to (0,0) = center of table.
            ensure_object_boundary_in_range:
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False:
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
            z_rotation:
                None: Add uniform random random z-rotation
                iterable (a,b): Uniformly randomize rotation angle between a and b (in radians)
                value: Add fixed angle z-rotation
        """
        self.x_range = x_range
        self.y_range = y_range
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.z_rotation = z_rotation
        self.np_random = np_random if np_random is not None else np.random


    def set_random_number_generator(self, np_random):
        self.np_random = np_random

    def set_ranges(self, x_range=None, y_range = None, z_rotation_range = None):
        if(x_range is not None):
            self.x_range = x_range
        if(y_range is not None):
            self.y_range = y_range

        if(z_rotation_range is not None):
            self.z_rotation = z_rotation_range

    def sample_obj_idx(self):
        return self.np_random.choice(self.n_obj)

    def sample_x(self, object_horizontal_radius):
        x_range = self.x_range
        if x_range is None:
            x_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(x_range)
        maximum = max(x_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.np_random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_horizontal_radius):
        y_range = self.y_range
        if y_range is None:
            y_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(y_range)
        maximum = max(y_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.np_random.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        if self.z_rotation is None or self.z_rotation is True:
            rot_angle = self.np_random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.z_rotation, collections.Iterable):
            rot_angle = self.np_random.uniform(
                high=max(self.z_rotation), low=min(self.z_rotation)
            )
        elif (isinstance(self.z_rotation, float) or isinstance(self.z_rotation, int)):
            rot_angle = self.z_rotation
        else:
            return [1, 0, 0, 0]

        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]

    def sample(self, push_object_index):
        pos_arr = []
        quat_arr = []
        placed_objects = []
        for index, obj_mjcf in enumerate(self.mujoco_objects):
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()

            if index != push_object_index:
                pos_arr.append(np.array([0.0, 0.0, 0.0]))
                quat_arr.append([1, 0, 0, 0])

            else:
                object_x = self.sample_x(horizontal_radius)
                object_y = self.sample_y(horizontal_radius)

                pos = (
                        self.table_top_offset
                        - bottom_offset
                        + np.array([object_x, object_y, 0])
                )
                placed_objects.append((object_x, object_y, horizontal_radius))
                # random z-rotation

                quat = self.sample_quat()

                quat_arr.append(quat)
                pos_arr.append(pos)

        return pos_arr, quat_arr
