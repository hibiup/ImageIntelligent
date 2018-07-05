from unittest import TestCase
from sift import sift_keypoint_sample as sift
import cv2
import matplotlib.pyplot as plt

print('OpenCV Version (should be 3.1.0 or higher, with nonfree packages installed, for this tutorial):')
print(cv2.__version__)


class TestSIFT(TestCase):
    def test_sift_keypoint(self):
        octo_front = cv2.imread('images/Octopus_Far_Front.jpg')
        octo_offset = cv2.imread('images/Octopus_Far_Offset.jpg')

        plt.imshow(sift.show_rgb_img(octo_offset))
        plt.show()
