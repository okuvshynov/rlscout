from utils.utils import parse_ids, split_int64
import unittest

class TestParseIds(unittest.TestCase):
    def test_empty(self):
        self.assertFalse(len(parse_ids("")))

    def test_single(self):
        ids = parse_ids("42")
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 42)

    def test_list(self):
        ids = parse_ids("42,53,64")
        self.assertEqual(len(ids), 3)
        for (a, b) in zip(ids, [42, 53, 64]):
            self.assertEqual(a, b)

    def test_range(self):
        ids = parse_ids("42-45")
        self.assertEqual(len(ids), 4)
        for (a, b) in zip(ids, [42, 43, 44, 45]):
            self.assertEqual(a, b)

    def test_rangelist(self):
        ids = parse_ids("3,42-45,500-501")
        self.assertEqual(len(ids), 7)
        for (a, b) in zip(ids, [3, 42, 43, 44, 45, 500, 501]):
            self.assertEqual(a, b)

    def test_overlap(self):
        ids = parse_ids("44,42-45,44-47")
        self.assertEqual(len(ids), 9)
        for (a, b) in zip(ids, [44, 42, 43, 44, 45, 44, 45, 46, 47]):
            self.assertEqual(a, b)

class TestSplitI64(unittest.TestCase):
    def test0(self):
        res = split_int64(-9223372036854775808)
        self.assertEqual(res, [0] * 8)

    def testFF(self):
        res = split_int64(9223372036854775807)
        self.assertEqual(res, [0xff] * 8)

if __name__ == '__main__':
    unittest.main()