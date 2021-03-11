import numpy as np
import cv2 as cv
import time
from disjoinset import DisjoinSet
from tqdm._tqdm import trange
from graph import Graph
from filter import GaussianFilter


def pre_processing(img, sigma):
    fltr = GaussianFilter(sigma)
    img = fltr.process(img)
    return img


class ImageSeg(object):
    def __init__(self, img, sigma, k):
        """

        :param img:
        :param sigma:
        :param k:
        """
        self.k = k
        self.img = pre_processing(img, sigma)
        self.vertexes_num = img.shape[0] * img.shape[1]
        self.G = Graph(self.img)

        self.regions = DisjoinSet(self.vertexes_num)
        self.int_dif = np.full(self.vertexes_num, k, dtype=np.float64)

    def segementation(self):
        self.segementRegion()
        img_seg = np.full(self.img.shape, -1, dtype=np.int32)
        color_matrix = np.random.randint(0, 255, (self.vertexes_num, 3), dtype=np.uint8)

        h, w, c = self.img.shape
        for i in range(self.vertexes_num):
            re = i
            x, y = re // w, re % w
            root_idx = self.regions.findRoot(re)
            color = color_matrix[root_idx, :]

            while all(img_seg[x, y, :] == [-1, -1, -1]):
                img_seg[x, y, :] = color

                p = self.regions.getParent(re)
                if re != p:
                    re = p
                    x, y = re // w, re % w
                else:
                    break

        # converting format
        img_seg = img_seg.astype(np.uint8)

        return img_seg

    def segementRegion(self):
        """
        separating the region
        :return:
        """
        self.sortEdges()

        print("union regions...")
        for i in trange(self.G.edges_num):
            edge = self.G.edges[i, :]
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

        idx = self.G.edges[0:self.G.edges_num, 2].argsort()

        # copy edge
        self.G.edges[0: self.G.edges_num, :] = self.G.edges[idx]

    def tau(self, region_size):
        """
        getting tau = k / region_size
        :param region_size:
        :return:
        """

        return self.k / region_size
