import cv2


def fetch_sift_info(img):
    gray_img = to_gray(img)
    kp, desc = gen_sift_features(gray_img)
    return gray_img, kp, desc


def to_gray(color_img):
    """ 转成灰阶图 """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # cv2.CV_32S
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


def inspect_keypoint_and_descriptor(keypoints, descriptors, index):
    kp = keypoints[index]
    desc = descriptors[index]

    print('\tangle:\t', kp.angle)
    print('\tclass_id:\t', kp.class_id)
    print('\toctave (image scale where feature is strongest):\t', kp.octave)
    print('\tpoint (x,y):\t', kp.pt)
    print('\tresponse:\t', kp.response)
    print('\tsize:\t', kp.size)
    print('\n\tdescriptor:\n', desc)


def show_img_keypoint(img):
    """ 返回带有 keypoint 的图 """
    gray_img, kp, desc = fetch_sift_info(img)

    # 合成带有 keypoint 的图
    kp_img = cv2.drawKeypoints(img, kp, img.copy())
    return kp_img, kp


def match_images(left_img, right_img, matches_lines):
    left_gray_img, left_kp, left_desc = fetch_sift_info(left_img)
    right_gray_img, right_kp, right_desc = fetch_sift_info(left_img)

    # create a BFMatcher object which will match up the SIFT features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(left_desc, right_desc)

    # Sort the matches in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # draw the top N matches
    N_MATCHES = matches_lines

    matched_img = cv2.drawMatches(
        left_img, left_kp,
        right_img, right_kp,
        matches[:N_MATCHES], right_img.copy(), flags=0)

    return matched_img
