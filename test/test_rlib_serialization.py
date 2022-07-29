import unittest

from chex import assert_equal

from reinforced_lib.utils.utils import ROOT_DIR

class TestRLibSerialization(unittest.TestCase):
    
    def test_simple(self):
        print("\ntest_simple_1")
        self.assertEqual(4, 2 + 4)
    
    pass

if __name__ == "__main__":
    unittest.main()