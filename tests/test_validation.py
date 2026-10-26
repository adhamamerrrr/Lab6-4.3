import unittest
from generate import generate_prompts
class ValidationTests(unittest.TestCase):
    def test_inputs_before_model_download(self):
        for prompts,tokens,temp in [([],30,.7),([''],30,.7),(['test'],0,.7),(['test'],30,0)]:
            with self.assertRaises(ValueError):generate_prompts('unused',prompts,tokens,temp)
