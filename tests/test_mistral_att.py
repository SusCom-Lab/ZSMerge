import unittest
from .test_utils import cls_init, gen_equal

class TestMistralGenerator(unittest.TestCase):
    model_name="mistralai/Mistral-7B-Instruct-v0.3"
    
    @classmethod
    def setUpClass(cls):
        cls_init(cls)
        return super().setUpClass()

    def test_gen(self):
        gen_equal(self)


if __name__ == "__main__":
    unittest.main()
