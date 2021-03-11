import numpy as np
from tqdm._tqdm import trange


class Graph(object):
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.c = img.shape
        self.edge_array_size = 4 * img.shape[0] * img.shape[1]
        self.edges_num = 0
        self.edges = np.zeros((self.edge_array_size, 3), dtype=np.float64)
        self.genGraph()

    def genGraph(self):
        idx = 0

        for x in trange(self.h):
            for y in range(self.w):
                for (i, j) in [(x, y + 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]:
                    if (0 <= i < self.h) and (0 <= j < self.w):
                        self.addEdge(idx, (x, y), (i, j))
                        idx += 1

    def addEdge(self, idx, adj_1, adj_2):
        self.edges[idx][0] = adj_1[0] * self.w + adj_1[1]
        self.edges[idx][1] = adj_2[0] * self.w + adj_2[1]
        self.edges[idx][2] = self.dif(adj_1, adj_2)
        self.edges_num += 1

    def getSpaceCoordination(self, x_y):
        coodination = [x_y[0], x_y[1], self.img[x_y[0], x_y[1], 0], self.img[x_y[0], x_y[1], 1], self.img[x_y[0], x_y[1], 2]]
        return np.array(coodination)

    def dif(self, adj_1, adj_2):
        adj_1 = self.getSpaceCoordination(adj_1)
        adj_2 = self.getSpaceCoordination(adj_2)
        return np.linalg.norm(adj_1 - adj_2, ord=2)
