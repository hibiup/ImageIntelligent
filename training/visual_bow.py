from sift import sift_feature_fetcher as sift
import aiofiles, asyncio
from functools import partial
import numpy
import cv2
import numpy as np
from sklearn import cluster
import pickle


def list_training_resource():
    with open("images/octopus/files.txt") as f:
        file_dict = dict(item.strip().split(" ") for item in f.readlines())
        return file_dict


def fetch_image_feature(raw_data, label):
    npimage = numpy.asarray(bytearray(raw_data), dtype=numpy.uint8)
    img = cv2.imdecode(npimage, -1)
    _, kp, features = sift.fetch_sift_info(img, sift.to_gray)
    return features, label


async def load_file_data(filename, label, loop, thread_pool):
    try:
        async with aiofiles.open(f"images/octopus/training/{filename}", "rb") as image:
            print(f'Opening: "{image._file.name}"')
            return await loop.run_in_executor(thread_pool, fetch_image_feature, await image.read(), label)
    except Exception as e:
        print(e)
        return None


def generate_raw_feature_dataset(thread_number=10):
    from concurrent import futures
    tasks = []
    file_dict = list_training_resource()

    loop = asyncio.get_event_loop()
    with futures.ThreadPoolExecutor(thread_number) as executor:
        __load_file_data = partial(load_file_data, loop=loop, thread_pool=executor)
        for filename, label in file_dict.items():
            tasks.append(asyncio.ensure_future(__load_file_data(filename, label)))

        loop.run_until_complete(asyncio.wait(tasks))

    raw_features = [task.result() for task in tasks if task.result() is not None]

    loop.close()
    return raw_features


def cluster_feature(description_set, cluster_model=cluster.KMeans(n_clusters=750, n_jobs=-1)):
    """
    用 K-Mean 算法为每个描述点 feature(descriptions) 分类。

    每幅图像都由不定数量的描述点（desc）组成，这些 desc 就类似文章中的“词汇”，但是即便是相似的“词汇”也可能完全不同，因此我们通过 K-Means
    算法寻找相识的描述点，并将它们归为同类，然后及于相同的标签，这样就归类了不同的 desc

    :param feature_set:
    :param cluster_model:
    :return:
    """
    model = cluster_model

    descriptors, labels = map(list, zip(*description_set))
    all_train_descriptors = [desc for desc_list in descriptors for desc in desc_list]

    print('Using clustering model %s...' % repr(model))
    # 训练 KMeans 模型．这一步的作用是为类似的　feature　归类．
    model.fit(all_train_descriptors)

    print('Using clustering model to generate BoW histograms for each image.')
    # 用 KMeans 模型来 “预测” 图像，意味着我们用归类过的feature来重新规范图像的含义，结果将用于训练 SVC 模型
    img_clustered_words = [model.predict(raw_words) for raw_words in descriptors]

    return model, zip(generate_histogram(img_clustered_words, model.n_clusters), labels)


def generate_histogram(data, minlength):
    """ 产生 BoW 直方图 """
    img_bow_hist = [np.bincount(clustered_words, minlength=minlength) for clustered_words in data]
    return img_bow_hist


def image_to_vector(img_path, cluster_model):
    """
    为要测试的图片生成直方图
    """
    img = cv2.imread(img_path)
    _, kp, desc = sift.fetch_sift_info(img)

    clustered_desc = cluster_model.predict(desc)
    test_img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # 因为 sklearn 接受二维参数，而　np.bincount　的输出是一维的，因此需要将结果转成　array[[1,2,3]]　的形式
    return test_img_bow_hist.reshape(1, -1)

