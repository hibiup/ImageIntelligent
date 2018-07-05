import cv2


def show_rgb_img(img):
    """ 返回带有keypoint 的灰阶图 """
    gray_img = to_gray(img)
    kp, desc = gen_sift_features(gray_img)
    # 合成带有 keypoint 的灰阶图
    kp_img = cv2.drawKeypoints(gray_img, kp, img.copy())
    return kp_img


def to_gray(color_img):
    """ 转成灰阶图 """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)   # cv2.CV_32S
    return gray


def gen_sift_features(gray_img):
    """ 生成 SIFT keypoint """
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc
