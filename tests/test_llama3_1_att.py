import unittest
from .test_utils import cls_init, gen_equal

class TestLlama3Generator(unittest.TestCase):
    model_name="meta-llama/Llama-3.1-8B-Instruct"
    
    @classmethod
    def setUpClass(cls):
        cls_init(cls)
        return super().setUpClass()

    def test_gen(self):
        gen_equal(self)


if __name__ == "__main__":
    unittest.main()
