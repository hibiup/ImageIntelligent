import cv2


def to_gray(color_img):
    """ 转成灰阶图 """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # cv2.CV_32S
    return gray


def to_canny(img, low_threshold=100, ratio=3):
    """ 抽取轮廓 """
    edges = cv2.Canny(img, low_threshold, ratio * low_threshold, cv2.COLOR_BGR2GRAY)
    return edges


def fetch_sift_info(img, gaussian=to_gray):
    gray_img = gaussian(img)
    kp, desc = gen_sift_features(gray_img)
    return gray_img, kp, desc


def gen_sift_features(gray_img):
    """
    生成 SIFT keypoint　和 descriptor

    keypoint: 关键点有三个信息，位置、尺度、方向。
    descriptor: 根据关键点周围 16×16 的像素区域，分成4个小块，每个小块创建 8 个bin 的直方图，这总共的 128 个信息的向量就是关键点描述符
                的主要内容。(images/keypoint_descriptor.pngs)。 descriptor 的作用是对模板图和实时图通过对比关键点描述符来判断两个
                关键点是否相同。使用的搜索算法为区域搜索算法当中最常用的k-d 树进行比对。比较之后消除错配点，最后得到两张图形是否一致。

    keypoint 和 descriptor 的算法说明参考:　https://blog.csdn.net/pi9nc/article/details/23302075
    """
    sift = cv2.xfeatures2d.SIFT_create()
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
    gray_img, kp, desc = fetch_sift_info(img, gaussian=to_gray)

    # 合成带有 keypoint 的图
    kp_img = cv2.drawKeypoints(img, kp, img.copy())
    return kp_img, kp


def match_images(left_img, right_img, gaussian=to_gray, lines=None, distance=0.7):
    """
    比对两组关键点描述，查找出相同的点位.

    distance: 两个特征向量之间的欧氏距离，越小表明匹配度越高

    参考：　https://www.cnblogs.com/wangguchangqing/p/4333873.html
    """
    left_gray_img, left_kp, left_desc = fetch_sift_info(left_img, gaussian)
    right_gray_img, right_kp, right_desc = fetch_sift_info(right_img, gaussian)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(left_desc, right_desc, k=2)   #matches = sorted(bf.match(left_desc, right_desc), key=lambda x: x.distance)

    good = []
    for m in matches:
        if 0 != m[1].distance and m[0].distance/m[1].distance < distance:      #if m.distance/250 < distance:   ???
            good.append(m)

    matched_img = cv2.drawMatchesKnn(                   ##matched_img = cv2.drawMatches(
        left_img, left_kp,
        right_img, right_kp,
        good if lines == None else good[:lines],        #matches[:lines],
        None,  #right_img.copy(),
        flags=2)

    return matched_img
