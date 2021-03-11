import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time

from filter import GaussianFilter
from graph import Graph
from seg import ImageSeg


# setting parse
parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", help="image needs to separate.", default="./image/tower.jpg")
parse.add_argument("-o", "--output", help="result image's name.", default="./result")
parse.add_argument("-s", "--sigma", help="the factor of GaussianFilter.", default=0.8)
parse.add_argument("-k", help="the factor to control the size of segment region.", default=800)
args = parse.parse_args()

# parameters
sigma = args.sigma
k = args.k
img_path = args.input
img_save_path = args.output

img_org = cv.imread(img_path, 1)
img_org = img_org[..., ::-1]

h, w, c = img_org.shape
vertexes_num = h * w

print("processing...")
fil = GaussianFilter(0.5)
img = fil.filter(img_org)
G = Graph(img)

seg = ImageSeg(vertexes_num, G.edges, G.edges_num, k)
seg.segment()

img_seg = np.full(img.shape, -1, dtype=np.int32)
color_matrix = np.random.randint(0, 255, (vertexes_num, 3), dtype=np.uint8)

for i in range(vertexes_num):
    re = i
    x, y = re // w, re % w
    root_idx = seg.regions.findRoot(re)
    color = color_matrix[root_idx, :]

    while all(img_seg[x, y, :] == [-1, -1, -1]):
        img_seg[x, y, :] = color

        p = seg.regions.getParent(re)
        if re != p:
            re = p
            x, y = re // w, re % w
        else:
            break

# converting storage format
img_seg = img_seg.astype(np.uint8)

# drawing the result
fig = plt.figure()
fig.suptitle('sigma={}, k={}'.format(sigma, k))
ax = fig.subplots(1, 2)

ax[0].imshow(img_org)
ax[0].set_title("Origin")

ax[1].imshow(img_seg)
ax[1].set_title("Segmented")

for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

# saving the result
img_seg_name = os.path.join(img_save_path, "seg_" + img_path.split('/')[-1])

if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)
plt_name = os.path.join(img_save_path, "plot_" + img_path.split('/')[-1])

img_save = cv.cvtColor(img_seg, cv.COLOR_RGB2BGR)
cv.imwrite(img_seg_name, img_save)
plt.savefig(plt_name, dpi=750, bbox_inches='tight')

print("segmentation have done.")
