import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import os
from seg import ImageSeg


# setting arguments
parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", help="image needs to separate.", default="./image/tower.jpg")
parse.add_argument("-o", "--output", help="result image's name.", default="./result")
parse.add_argument("-s", "--sigma", help="the factor of GaussianFilter.", default=0.8)
parse.add_argument("-k", help="the factor to control the size of segment region.", default=800)
args = parse.parse_args()

# parsing arguments
sigma = args.sigma
k = args.k
img_path = args.input
img_save_path = args.output

# loading ...
print("loading...")
img = cv.imread(img_path, 1)
img = img[..., ::-1]

# segementing
print("processing...")
seg = ImageSeg(img, sigma, k)
img_seg = seg.segementation()

# drawing the result
fig = plt.figure()
fig.suptitle('sigma={}, k={}'.format(sigma, k))
ax = fig.subplots(1, 2)

ax[0].imshow(img)
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
