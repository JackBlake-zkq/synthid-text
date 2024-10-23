# SynthID Text

This repository provides a reference implementation of the SynthID Text
watermarking and detection capabilities, it is not intended for production use.
The core library is [distributed on PyPI][synthid-pypi] for easy installation in
the [Python Notebook example][synthid-colab], which demonstrates how to apply
these tools with the [Gemma][gemma] and [GPT-2][gpt2] models.

## Installation

From the `synthid_text` directory run following steps:

```shell
python3 -m venv ~/.venvs/synthid
source ~/.venvs/synthid/bin/activate
pip install -r requirements.txt
pip install .
pip install notebook
python -m notebook
```

Once your kernel is running navigate to .pynb file to execute.

## Usage

The [Colab Notebook][synthid-colab] is self-contained reference implementation
that:

1.  Extends the [`GemmaForCausalLM`][transformers-gemma] and
    [`GPT2LMHeadModel`][transformers-gpt2] classes from
    [Hugging Face Transformers][transformers] with a [mix-in][synthid-mixin] to
    enable watermarking text content generated by models running in
    [PyTorch][pytorch]; and
1.  Detects the watermark. This can be done either with the simple [Mean
    detector][synthid-detector-mean] which requires no training, or with the
    more powerful [Bayesian detector][synthid-detector-bayesian] that requires
    [training][synthid-detector-trainer].

The notebook is designed to be run end-to-end with either a Gemma or GPT-2
model, and runs best on the following runtime hardware, some of which may
require a [Colab Subscription][colab-subscriptions].

*   Gemma v1.0 2B IT: Use a GPU with 16GB of memory, such as a T4.
*   Gemma v1.0 7B IT: Use a GPU with 32GB of memory, such as an A100.
*   GPT-2: Any runtime will work, though a High-RAM CPU or any GPU will be
    faster.

NOTE: This implementation is for reference and research reproducibility purposes
only. Due to minor variations in Gemma and Mistral models across implementations,
we expect minor fluctuations in the detectability and perplexity results obtained
from this repository versus those reported in the paper. The subclasses
introduced herein are not designed to be used in production systems. Check out
the official SynthID logits processor in [Hugging Face Transformers][transformers]
for a production-ready implementation.

NOTE: The `synthid_text.hashing_function.accumulate_hash()` function, used while
computing G values in this reference implementation, does not provide any
guarantees of cryptographic security.

### Defining a watermark configuration

SynthID Text produces unique watermarks given a configuration, with the most
important piece of these configurations being the `keys`: a sequence of unique
integers where `len(keys)` corresponds to the number of layers in the
watermarking or detection models.

The structure of a configuration is described in the following `TypedDict`
subclass, though in practice, the [mixin][synthid-mixin] class in this library
uses a static configuration.

```python
from collections.abc import Sequence
from typing import TypedDict

import torch


class WatermarkingConfig(TypedDict):
    ngram_len: int
    keys: Sequence[int]
    sampling_table_size: int
    sampling_table_seed: int
    context_history_size: int
    device: torch.device
```

### Applying a watermark

Watermarks are applied by a [mix-in][synthid-mixin] class that wraps the
[`GemmaForCausalLM`][transformers-gemma] and
[`GPT2LMHeadModel`][transformers-gpt2] classes from Transformers, which results
in two subclasses with the same API that you are used to from Transformers.
Remember that the mix-in provided by this library uses a static watermarking
configuration, making it unsuitable for production use.

```python
from synthid_text import synthid_mixin
import transformers
import torch


DEVICE = (
    torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
INPUTS = [
    "I enjoy walking with my cute dog",
    "I am from New York",
    "The test was not so very hard after all",
    "I don't think they can score twice in so short a time",
]
MODEL_NAME = 'google/gemma-2b-it'
TEMPERATURE = 0.5
TOP_K = 40
TOP_P = 0.99

# Initialize a standard tokenizer from Transformers.
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
# Initialize a a SynthID Text-enabled model.
model = synthid_mixin.SynthIDGemmaForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
# Prepare your inputs in the usual way.
inputs = tokenizer(
    INPUTS,
    return_tensors='pt',
    padding=True,
).to(DEVICE)
# Genreate watermarked text.
outputs = model.generate(
    **inputs,
    do_sample=True,
    max_length=1024,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
)
```

### Detecting a watermark

Watermark detection can be done using a variety of scoring functions (see
paper). This repository contains code for the Mean, Weighted Mean, and Bayesian
scoring functions described in the paper. The colab contains examples for how
to use these scoring functions.

The Bayesian detector must be trained on watermarked and unwatermarked data
before it can be used. The Bayesian detector must be trained for each unique
watermarking key, and the training data used for this detector model should be
independent from, but representative of the expected character and quality of
the text content the system will generate in production.

```python
import jax.numpy as jnp
from synthid_text import train_detector_bayesian


def load_data():
  # Get your training and test data into the system.
  pass


def process_training_data(split):
  # Get the G values, masks, and labels for the provided split.
  pass


train_split, test_split = load_data()
train_g_values, train_masks, train_labels = process_training_data(train_split)
test_g_values, test_masks, test_labels = process_training_data(test_split)

detector, loss = train_detector_bayesian.optimize_model(
    jnp.squeeze(train_g_values),
    jnp.squeeze(train_masks),
    jnp.squeeze(train_labels),
    jnp.squeeze(test_g_values),
    jnp.squeeze(test_masks),
    jnp.squeeze(test_labels),
)
```

Once the Bayesian detector is trained, use the `detector.score()` function to
generate a per-example score indicating if the text was generated with the given
watermarking configuration. Score values will be between 0 and 1, with scores
closer to 1 indicating higher likelihood that the text was generated with the
given watermark. You can adjust the acceptance threshold to your needs.

```python
from synthid_text import logits_processing


CONFIG = synthid_mixin.DEFAULT_WATERMARKING_CONFIG

logits_processor = logits_processing.SynthIDLogitsProcessor(
    **CONFIG, top_k=TOP_K, temperature=TEMPERATURE
)

# Get only the generated text from the models predictions.
outputs = outputs[:, inputs_len:]

# Copute the end-of-sequence mask, skipping first ngram_len - 1 tokens
# <bool>[batch_size, output_len]
eos_token_mask = logits_processor.compute_eos_token_mask(
    input_ids=outputs,
    eos_token_id=tokenizer.eos_token_id,
)[:, CONFIG['ngram_len'] - 1 :]
# Compute the context repetition mask
# <bool>[batch_size, output_len - (ngram_len - 1)]
context_repetition_mask = logits_processor.compute_context_repetition_mask(
    input_ids=outputs
)

# Compute the mask that isolates the generated text.
combined_mask = context_repetition_mask * eos_token_mask
# Compute the G values for the generated text.
g_values = logits_processor.compute_g_values(input_ids=outputs)

# Score the G values, given the combined mask, and output a per-example score
# indicating whether the
detector.score(g_values.cpu().numpy(), combined_mask.cpu().numpy())
```

## Running the tests

```shell
python3 -m venv ~/.venv/synthid_text
source ~/.venv/synthid_text/bin/activate
pip install -r requirements.txt
pytest .
```


## Human Data

We release the human evaluation data, where we compare watermarked text against unwatermarked text generated from the Gemma 7B model.
The data is located in `data/human_eval.jsonl`.
To get the prompts used for generating the responses, please use the following code.

```
import json
import tensorflow_datasets as tfds

ds = tfds.load('huggingface:eli5/LFQA_reddit', split='test_eli5')
id_to_prompt = {}
for x in ds.as_numpy_iterator():
  id_to_prompt[x['q_id'].decode()] = x['title'].decode()

full_data = []
with open('./data/human_eval.jsonl') as f:
  for json_str in f:
    x = json.loads(json_str)
    x['question'] = id_to_prompt[x['q_id']]
    full_data.append(x)
```



## Citing this work

The relevant paper is currently under review, during which time this repository
is private. Once it goes public, a bibtex reference will be provided here.

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

[colab-subscriptions]: https://colab.research.google.com/signup
[flax]: https://github.com/google/flax
[gemma]: https://ai.google.dev/gemma/docs/model_card
[gpt2]: https://huggingface.co/openai-community/gpt2
[jax]: https://github.com/google/jax
[pytorch]: https://pytorch.org/
[synthid-colab]: https://colab.research.google.com/github/google-deepmind/synthid-text/blob/main/notebooks/synthid_text_huggingface_integration.ipynb
[synthid-pypi]: https://pypi.org/project/synthid-text/
[synthid-detector-bayesian]: ./src/synthid_text/detector_bayesian.py
[synthid-detector-mean]: ./src/synthid_text/detector_mean.py
[synthid-detector-trainer]: ./src/synthid_text/train_detector_bayesian.py
[synthid-mixin]: ./src/synthid_text/synthid_mixin.py
[transformers]: https://github.com/huggingface/transformers
[transformers-gemma]: https://github.com/huggingface/transformers/blob/e55b33ceb4b0ba3c8c11f20b6e8d6ca4b48246d4/src/transformers/models/gemma/modeling_gemma.py#L996
[transformers-gpt2]: https://github.com/huggingface/transformers/blob/e55b33ceb4b0ba3c8c11f20b6e8d6ca4b48246d4/src/transformers/models/gpt2/modeling_gpt2.py#L1185
