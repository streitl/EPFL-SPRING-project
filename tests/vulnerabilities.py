import unittest

import pandas as pd
import numpy as np

import src.vulnerabilities as vul
from src.models import SRR
from src.preprocessing import one_hot_encode, bin_features

def mock_dataset():
    X = pd.DataFrame(columns=['humidity', 'wind_speed'])
    X['humidity'] =   pd.Series([  5,   1,  10,   2,  16,  12,   3,  17,   1,   1,  20,  16])
    X['wind_speed'] = pd.Series(['H', 'M', 'M', 'M', 'M', 'S', 'S', 'S', 'S', 'H', 'H', 'H'])
    y =               pd.Series([  1,   0,   1,   1,   0,   0,   0,   1,   0,   1,   1,   0])

    X_binned, _ = bin_features(X, X, nbins=3)
    return X_binned, y


class TestVulnerabilities(unittest.TestCase):

    def test_find_adversarial_examples(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)
        print(model)

        adversaries = vul.find_adversarial_examples(model, X, y, can_change=['wind_speed'], unit_changes=True)

        self.assertGreater(adversaries.shape[0], 0)


    def test_binned_features_pass_monotonicity_no_binned_features(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        self.assertTrue(vul.binned_features_pass_monotonicity(model, X, y))


    def test_binned_features_pass_monotonicity(self):
        X, y = mock_dataset()
        removals = [1, 3, 6, 10]
        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X.drop(removals)), y.drop(removals))

        self.assertTrue(vul.binned_features_pass_monotonicity(model, X, y))


    def test_poisoning_attack_flip_sign_fails(self):
        X, y = mock_dataset()

        X_binned, _ = bin_features(X, X, nbins=3)

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X_binned), y)

        self.assertRaises(ValueError, vul.poisoning_attack, model, X_binned, y, 'wind_speed', 'H', goal='flip_sign')
        self.assertRaises(ValueError, vul.poisoning_attack, model, X_binned, y, 'wind_speed', 'S', goal='flip_sign')


    def test_poisoning_attack_nullify(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        for cat in ['H', 'S']:
            removals = vul.poisoning_attack(model, X, y, 'wind_speed', cat, goal='nullify', use_stats=True)

            poisoned = SRR.copy_params(model)
            poisoned.fit(one_hot_encode(X.drop(removals)), y.drop(removals))

            self.assertEqual(poisoned.get_weight('wind_speed', cat), 0)


    def test_poisoning_attack_remove_feature(self):
        X, y = mock_dataset()

        model = SRR(k=1, M=2, cv=2, Cs=100)
        model.fit(one_hot_encode(X), y)

        removals = vul.poisoning_attack(model, X, y, 'wind_speed', goal='remove_feature')

        poisoned = SRR.copy_params(model)
        poisoned.fit(one_hot_encode(X.drop(removals)), y.drop(removals))

        self.assertTrue(np.isnan(poisoned.get_weight('wind_speed', '')))



if __name__ == '__main__':
    unittest.main()