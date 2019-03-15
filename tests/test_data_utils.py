import unittest

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
sys.path.append(os.path.join(os.getcwd(), "solver"))

from data_utils import *

class TestMain(unittest.TestCase):

    def test_get_zero_string(self):

        self.assertEqual(get_zero_string(1),"001")
        self.assertEqual(get_zero_string(10),"010")
        self.assertEqual(get_zero_string(0),"000")
        self.assertEqual(get_zero_string(120),"120")
        self.assertEqual(get_zero_string(990),"990")
        self.assertEqual(get_zero_string(100),"100")

if __name__ == '__main__':
    unittest.main()
