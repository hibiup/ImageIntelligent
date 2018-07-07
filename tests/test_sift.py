from unittest import TestCase
from sift import sift_keypoint_sample as sift
import cv2
import matplotlib.pyplot as plt
from numpy import arange, cos, sin, pi

print('OpenCV Version (should be 3.1.0 or higher, with nonfree packages installed, for this tutorial):')
print(cv2.__version__)


def get_sample_images():
    octo_front = cv2.imread('images/Octopus_Far_Front.jpg')     # Octopus_carton.jpg
    octo_offset = cv2.imread('images/Octopus_Far_Offset.jpg')   # Octopus_in_the_sea.jpg
    return octo_front, octo_offset


def draw_circle(kp):
    def generate_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        return x, y

    for p in kp:
        x, y = generate_circle(p.pt, p.size)
        plt.plot(x, y, 'b', linewidth=2)


class TestSIFT(TestCase):

    def test_canny(self):
        octo_front, _ = get_sample_images()
        edges_img = sift.to_canny(octo_front)
        plt.imshow(edges_img)
        plt.show()

    def test_sift_keypoint(self):
        octo_front, octo_offset = get_sample_images()

        im, kp = sift.show_img_keypoint(octo_offset)
        plt.imshow(im)
        draw_circle(kp)

        plt.show()

    def test_image_matching(self):
        octo_front, octo_offset = get_sample_images()

        plt.imshow(sift.match_images(octo_front, octo_offset, gaussian=sift.to_gray))
        plt.show()

    def test_show_desc(self):
        """ 显示 4 对 descriptor 对比图 """

        octo_front, octo_offset = get_sample_images()

        gray_front_img, front_kp, front_desc = sift.fetch_sift_info(octo_front)
        gray_offset_img, offset_kp, offset_desc = sift.fetch_sift_info(octo_offset)
        sift.inspect_keypoint_and_descriptor(front_kp, front_desc, 0)

        index = 0
        columns = 4
        rows = 2
        fig = plt.figure(frameon=False)
        for i in range(1, columns*rows + 1):
            fig.add_subplot(rows, columns, i)
            desc_img = front_desc[index + i] if 1 == i % 2 else offset_desc[index + i - 1]
            plt.imshow(desc_img.reshape(16, 8), interpolation='none')

        plt.show()
