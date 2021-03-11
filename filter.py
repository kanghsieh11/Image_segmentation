import numpy as np


class GaussianFilter(object):
    def __init__(self, sigma, k_size=3):
        """

        :param sigma: standard deviation of gaussian distribution
        :param k_size: kernel size of Filter
        """

        self.sigma = max(sigma, 0.01)
        self.kernel_size = k_size
        self.kernel = self.genKernel()

    def __repr__(self):
        return "GaussianFilter with sigma {0}, kernel size {1}".format(self.sigma, self.kernel_size)

    def setSigma(self, sigma):
        self.sigma = max(sigma, 0.01)

    def genKernel(self):
        """
        the function is used to generate the kernel of filter
        :return:
        """
        K = np.zeros(self.kernel_size, dtype=np.float64)
        for i in range(self.kernel_size):
            K[i] = np.exp(-0.5 * ((i - self.kernel_size // 2) / self.sigma) ** 2)
        K = K / K.sum()
        return K

    def filter(self, img):
        """
        Gaussian filter can be divided into two steps:
        first, convolute the image with one-dimensional Gaussian function in the horizontal/vertical
        direction, then convolute the convoluted result in the vertical/horizontal direction using
        the same one-dimensional Gaussian function template.

        :param img:
        :return:
        """
        img_row_filtered = self.separateFilter(img)
        img_filterd = self.separateFilter(img_row_filtered)
        return img_filterd

    def separateFilter(self, img):
        """
        Gaussian filtering is performed along the horizontal direction of the image. Note that the
        image will be transposed after Gaussian filtering.
        :param img:
        :return:
        """
        assert(img.shape[2] == 1 or img.shape[2] == 3)

        img_out = np.zeros_like(img, dtype=np.float64)
        h, w, c = img.shape

        for z in range(c):
            for x in range(h):
                for y in range(w):
                    s = 0
                    for i in range(self.kernel_size):
                        if (y + i - self.kernel_size // 2) in range(w):
                            s += self.kernel[i] * img[x, y + i - self.kernel_size // 2, z]
                    img_out[x, y, z] = s
        return img_out.transpose([1, 0, 2])
