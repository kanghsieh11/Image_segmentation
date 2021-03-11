import numpy as np
import cv2 as cv
import time
from disjoinset import DisjoinSet
from tqdm._tqdm import trange


class ImageSeg(object):
    def __init__(self, vertexes_num, edges, edges_num, k):
        self.vertexes_num = vertexes_num
        self.edges = edges
        self.edges_num = edges_num
        self.k = k
        self.regions = DisjoinSet(vertexes_num)
        self.int_dif = np.full(vertexes_num, k, dtype=np.float64)

    def segment(self):
        """
        separating the image
        :return:
        """
        self.sortEdges()

        print("union regions...")
        for i in trange(self.edges_num):
            edge = self.edges[i, :]
            r_1 = self.regions.findRoot(edge[0])
            r_2 = self.regions.findRoot(edge[1])

            if r_1 != r_2:
                if edge[2] <= min(self.int_dif[r_1], self.int_dif[r_2]):
                    self.regions.union(r_1, r_2)
                    root_idx = self.regions.findRoot(r_1)
                    self.int_dif[root_idx] = edge[2] + self.tau(self.regions.getSetSize(root_idx))

    def sortEdges(self):
        """
        sorting edge by dif
        """

        idx = self.edges[0:self.edges_num, 2].argsort()

        # copy edge
        self.edges[0: self.edges_num, :] = self.edges[idx]

    def tau(self, region_size):
        """
        getting tau = k / region_size
        :param region_size:
        :return:
        """

        return self.k / region_size
