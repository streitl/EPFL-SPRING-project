import unittest

import src.loader as loader

uci_datasets = ['adult', 'annealing', 'audiology-std', 'bank', 'bankruptcy', 'car',
                    'chess-krvk', 'chess-krvkp', 'congress-voting', 'contrac', 'credit-approval',
                    'ctg', 'cylinder-bands', 'dermatology', 'german_credit', 'heart-cleveland',
                    'ilpd', 'mammo', 'mushroom', 'wine', 'wine_qual']

all_datasets = uci_datasets + ['texas', 'ieeecis']


class TestLoader(unittest.TestCase):

    def test_load_all_uci(self):
        for name in uci_datasets:
            X, y = loader.load_dataset(name)
            self.assertEqual(X.shape[0], y.shape[0], 'X and y do not have the same number of elements')

    def test_load_texas(self):
        X, y = loader.load_texas()
        self.assertEqual(X.shape[0], y.shape[0], 'X and y do not have the same number of elements')

    def test_load_ieeecis(self):
        X, y = loader.load_ieeecis()
        self.assertEqual(X.shape[0], y.shape[0], 'X and y do not have the same number of elements')

    def test_load_nonexistent(self):
        self.assertRaises(ValueError, loader.load_dataset, 'this_dataset_does_not_exist')


if __name__ == '__main__':
    unittest.main()
