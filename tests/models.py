import unittest

import pandas as pd

import src.models as models
from src.preprocessing import one_hot_encode

def mock_dataset():
    X = pd.DataFrame(columns=['height', 'weight'])
    X['height'] = pd.Series(['L', 'M', 'L', 'S', 'M'] * 10)
    X['weight'] = pd.Series(['H', 'L', 'M', 'M', 'H'] * 10)
    y = pd.Series([1, 0, 1, 0, 0] * 10)
    return X, y

class TestSRRMethods(unittest.TestCase):

    def test_train_simple(self):
        X, y = mock_dataset()

        model = models.SRR(k=1, M=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        self.assertIn('height', model.selected_features)
        self.assertGreater(model.get_weight('height', 'L'), 0)

        self.assertEqual(set(model.df.index.levels[0]) - {'bias'}, set(model.selected_features))


    def test_save_load(self):
        X, y = mock_dataset()

        model = models.SRR(k=1, M=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        model.save('test_save_load')
        model_new = models.SRR.load('test_save_load', k=1, M=2)

        # Check whether the saved and loaded models are the same
        self.assertEqual(model.k, model_new.k)
        self.assertEqual(model.M, model_new.M)
        self.assertEqual(model.cv, model_new.cv)
        self.assertEqual(model.Cs, model_new.Cs)
        self.assertEqual(model.n_jobs, model_new.n_jobs)
        self.assertEqual(model.max_iter, model_new.max_iter)
        self.assertEqual(model.random_state, model_new.random_state)

        self.assertEqual(model.selected_features, model_new.selected_features)
        self.assertTrue((model.df == model_new.df).all().all())


    def test_predict(self):
        X, y = mock_dataset()

        model = models.SRR(k=1, M=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        y_pred = model.predict(one_hot_encode(X))

        # This dataset is perfectly separable so should have 100% accuracy
        self.assertTrue((y == y_pred).all())


if __name__ == '__main__':
    unittest.main()
