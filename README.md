# Lab 6: GPT-Neo prompt comparison

The included PDF contains the original basic and enhanced GPT-Neo examples. `generate.py` is a new refactor of those visible examples: it loads the model once, passes the attention mask, validates context length and uses `max_new_tokens` to separate the output budget from the prompt.

```sh
python -m pip install -r requirements.txt
python generate.py
python generate.py --model EleutherAI/gpt-neo-1.3B --prompt "List the top three uses of apple fruit."
python -m unittest discover -s tests -v
```

The default 125M model is smaller than the PDF's 1.3B model. First use downloads model weights from Hugging Face; no weights or generated results are committed. This is a base text completion model. Input-validation tests run without downloading a model; neural inference was not run during this update. Seeded output is not guaranteed identical across devices or library versions.
