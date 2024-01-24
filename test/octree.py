from numba import jit
import numpy as np
from collections import deque

class AABB:
    MaxCount = 50
    def __init__(self, point1, point2):
        """Points must be np.ndarrays."""
        assert all(point1 > point2)

        self.point1 = point1
        self.point2 = point2

        self.satellite_coords = []
        self.satellite_indices = []
        self.is_leaf = True
        self.children : list[AABB] = []
    
    def contains_point(self, point):
        return all((self.point1 > point) & (point > self.point2))
    
    def add_point(self, index, point):

        self.satellite_coords.append(point)
        self.satellite_indices.append(index)
        AABB_dict[index] = self

        if len(self.satellite_coords) > AABB.MaxCount:
            self.partition()
    
    def partition(self):
        self.is_leaf = False
        partition_size = (self.point1 - self.point2) * 0.5

        for x_offset in (0, partition_size[0]):
            for y_offset in (0, partition_size[1]):
                for z_offset in (0, partition_size[2]):
                    new_point1 = self.point1 - [x_offset, y_offset, z_offset]
                    self.children.append(AABB(new_point1, new_point1 - partition_size))
        
        for index, coords in zip(self.satellite_indices, self.satellite_coords):
            for child in self.children:
                if child.contains_point(coords):
                    child.add_point(index, coords)
                    break
                
        self.satellite_coords.clear()


from scipy.spatial.distance import euclidean

AABB_dict = dict()

def generate_octree(satellite_coords, root_bucket_size):
    root_bucket_size = np.array(root_bucket_size)
    root_bucket = AABB(root_bucket_size*0.5, -root_bucket_size*0.5)

    for index, coords in enumerate(satellite_coords):
        coords = np.array(coords)

        if root_bucket.is_leaf:
            root_bucket.add_point(index, coords)
        else:
            queue = deque()
            queue.append(root_bucket)
            while len(queue) > 0:
                bucket : AABB = queue.popleft()
                
                if bucket.contains_point(coords):
                    if not bucket.is_leaf:
                        for child in bucket.children:
                            queue.append(child)        
                    else:
                        bucket.add_point(index, coords)
                        break
    
    min_distances = []
    for index, coords in enumerate(satellite_coords):
        distances = []
        bucket = AABB_dict[index]
        for other_index, other_coords in zip(bucket.satellite_indices, bucket.satellite_coords):
            if index != other_index:
                distances.append(euclidean(coords, other_coords))
        if len(distances) > 0:
            min_distances.append(np.min(distances))
    
    print(np.min(min_distances))



