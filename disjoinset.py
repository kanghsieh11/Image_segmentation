import numpy as np


class DisjoinSet(object):
    def __init__(self, nodes_num):
        """
        using union-find sets to realize MST forest
        :param nodes_num: the number of tree nodes(that is, the number of pixels).
        """
        self.sets_num = nodes_num
        # tree:  idx  parent  set_size
        self.tree = np.zeros((nodes_num, 3), dtype=np.int)
        for i in range(nodes_num):
            self.tree[i, 0] = i  # init parent
            self.tree[i, 1] = 1  # init size

    def getSetSize(self, idx):
        return self.tree[idx, 1]

    def findRoot(self, idx):
        idx = int(idx)
        while idx != self.tree[idx, 0]:
            idx = self.tree[idx, 0]
        return idx

    def isConneted(self, x, y):
        x = int(x)
        y = int(y)
        return self.findRoot(x) == self.findRoot(y)

    def getParent(self, idx):
        return self.tree[idx, 0]

    def union(self, x, y):
        if self.tree[x, 2] > self.tree[y, 2]:
            self.tree[y, 0] = x
            self.tree[x, 1] += self.tree[y, 1]
        else:
            self.tree[x, 0] = y
            self.tree[y, 1] += self.tree[x, 1]
            if self.tree[x, 2] == self.tree[x, 2]:
                self.tree[y, 2] += 1
            self.sets_num -= 1
