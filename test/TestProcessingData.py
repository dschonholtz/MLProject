"""
Tests the functionality of the Preprocessor module
"""
import os

from Preprocess import H5Dataset
import unittest
from glob import glob


# write a test for the filename sort key. It takes in a filename and checks if
# the value at the end of it before the extension is captured correctly
class TestPreprocessor(unittest.TestCase):
    def test_filename_sort_key(self):
        filename = "test_sample1.h5"
        self.assertEqual(H5Dataset.filename_sort_key(filename), 1)
        filename = "sample_sample286.h5"
        self.assertEqual(H5Dataset.filename_sort_key(filename), 286)
        filename = "sample_123_abc_bcd34_sample24.h5"
        self.assertEqual(H5Dataset.filename_sort_key(filename), 24)

    def test_read_and_normalize_h5(self):
        # get a single h5 from this dir: /mnt/wines/intra/original_data/inv/pat_26402/downsampled_2/ieeg
        dataset = H5Dataset('test_data')
        print(len(dataset.x))
        print(dataset.x[1].shape)
        print(dataset[0][1])

    def test_actual_data(self):
        my_path = '/mnt/wines/intra/original_data/inv/pat_26402' + os.sep + 'downsampled_2' + os.sep + 'ieeg'
        dataset = H5Dataset(my_path)
        # print(dataset)


    # def test_dir_with_least_h5(self):
    #     self.get_dir_with_least_h5_files()

    def get_dir_with_least_h5_files(self):
        # Searches through all of the files at '/mnt/wines/intra/original_data/inv/pat_*/downsampled_2/ieeg'
        # to see which dir has the least files in it. returns the dir name.
        # the dir name is the patient number where the * is replaced by the number
        # of the patient
        # get all dirs that match the pattern:
        all_dirs_and_files = glob('/mnt/wines/intra/original_data/inv/pat_*')
        # get all dirs only
        all_dirs = [dir for dir in all_dirs_and_files if os.path.isdir(dir)]
        min_count = 100000
        min_dir = None
        for d in all_dirs:
            all_h5s = glob(d + '/downsampled_2/ieeg/*.h5')
            print(f'One h5 file in {d} is {all_h5s[0]}')
            count = len(all_h5s)
            if count < min_count:
                min_count = count
                min_dir = d
        print(min_dir)
        print(min_count)
        return min_dir





