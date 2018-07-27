from unittest import TestCase
from training import visual_bow, train_svc
import pickle
import matplotlib.pyplot as plt
import numpy as np


class TestImageRecognize(TestCase):
    def test_generate_image_histogram(self):
        with open("data/cluster_model.pickle", 'rb') as f:
            model = pickle.load(f)
            histogram = visual_bow.image_to_vector("images/Octopus_Far_Front.jpg", cluster_model=model)
            print(histogram)

            plt.imshow(histogram, interpolation='none')
            plt.show()

    def test_generate_training_raw_feature(self):
        """
        Generate training raw data
        :return: Raw training dataset
        """
        fs = visual_bow.generate_raw_feature_dataset(sample=100)

        with open("data/raw_features.pickle", 'wb+') as f:
            pickle.dump(fs, f)

    def test_cluster_features(self):
        """
        Cluster raw data
        :return: Clustered dataset
        """
        from sklearn import cluster

        with open("data/raw_features.pickle", 'rb') as f:
            fs = pickle.load(f)
            model, histogram = visual_bow.cluster_feature(fs)

            with open("data/cluster_model.pickle", 'wb+') as cf, open("data/histogram_dataset.pickle", 'wb+') as pf:
                pickle.dump(model, cf)
                pickle.dump(histogram, pf)

    def test_train_svc(self):
        with open("data/histogram_dataset.pickle", 'rb') as f:
            hist_dataset = pickle.load(f)
            X, y = zip(*hist_dataset)
            svc = train_svc.train_svc(np.array(X), np.array(y))

            with open("data/svc_model.pickle", "wb+") as svcfile:
                pickle.dump(svc, svcfile)

    def test_predict(self):
        for factor in ["test"]:
            print(f"== {factor} =========================")
            test_files = visual_bow.list_training_resource(factor)
            with open("data/svc_model.pickle", "rb") as model_file, open("data/cluster_model.pickle", 'rb') as cluster_file:
                model = pickle.load(model_file)
                cluster = pickle.load(cluster_file)
                for file, label in test_files.items():
                    vector = visual_bow.image_to_vector("images/trainingset/" + file, cluster_model=cluster)
                    if vector is not None:
                        print(file, label.strip(), model.predict(vector))
