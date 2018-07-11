from unittest import TestCase
from training import training


class TestImageIntelligent(TestCase):
    def test_generate_training_data(self):
        trainer = training.Training()
        trainer.generate_training_data()
