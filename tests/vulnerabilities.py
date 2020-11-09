import unittest

import pandas as pd

import src.vulnerabilities as vul
from src.models import SRR
from src.preprocessing import one_hot_encode, bin_features

def mock_dataset():
    X = pd.DataFrame(columns=['humidity', 'wind_speed'])
    X['humidity'] =   pd.Series([  1, 0.1,  32, 9.3, 2.3, 1.2,   6, 7.5,   2,  21])
    X['wind_speed'] = pd.Series(['S', 'H', 'M', 'M', 'S', 'M', 'S', 'S', 'H', 'M'])
    y =               pd.Series([  1,   0,   0,   0,   1,   0,   0,   1,   1,   0])

    X_binned, _ = bin_features(X, X, nbins=3)
    return X_binned, y


class TestVulnerabilities(unittest.TestCase):

    def test_find_adversarial_examples(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        adversaries = vul.find_adversarial_examples(model, X, y, can_change=['wind_speed'], unit_changes=True)

        self.assertGreater(adversaries.shape[0], 0)

    def test_binned_features_pass_monotonicity_no_binned_features(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        self.assertTrue(vul.binned_features_pass_monotonicity(model, X, y))


    def test_binned_features_pass_monotonicity(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X.drop([6, 9, 5, 7])), y.drop([6, 9, 5, 7]))

        self.assertFalse(vul.binned_features_pass_monotonicity(model, X, y))


    def test_poisoning_attack_flip_sign(self):
        X, y = mock_dataset()

        X_binned, _ = bin_features(X, X, nbins=3)

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X_binned), y)

        removals = vul.poisoning_attack_flip_sign(model, X_binned, y, 'wind_speed', 'S')
        self.assertEqual(removals, [0, 9, 5, 1, 2])

        removals = vul.poisoning_attack_flip_sign(model, X_binned, y, 'wind_speed', 'M')
        self.assertEqual(removals, [4, 0, 1, 8])


    def test_poisoning_attack_nullify(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        removals = vul.poisoning_attack_nullify(model, X, y, 'wind_speed', 'S')
        self.assertEqual(removals, [1])

        removals = vul.poisoning_attack_nullify(model, X, y, 'wind_speed', 'M')
        self.assertEqual(removals, [4])


    def test_poisoning_attack_remove_feature(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        removals = vul.poisoning_attack_remove_feature(model, X, y, 'wind_speed')
        self.assertEqual(removals, [6, 9, 5, 7])


if __name__ == '__main__':
    unittest.main()