# NanoGPT

A compact character level language model project built from scratch with PyTorch. It starts with a bigram baseline and then implements a small GPT-style transformer.

## How it works

Both models read `input.txt`, build a character vocabulary, and split the encoded text into training and validation data. Random context windows are used to predict the next character. After training, each model generates new text one character at a time.

## Models

| Script | Model | Main ideas |
| --- | --- | --- |
| `biagram.py` | Bigram language model | Token embeddings and direct next-character prediction |
| `gpt.py` | Transformer language model | Positional embeddings, causal self-attention, multiple heads, feed-forward layers, residual connections, and layer normalization |

## Files

- `input.txt` contains the training corpus.
- `biagram.py` trains and samples from the bigram baseline.
- `gpt.py` trains and samples from the transformer.
- `out.txt` can store training logs and generated text. It is ignored by Git because it is generated output.

## Requirements

- Python 3
- PyTorch

CUDA is selected automatically when available. Otherwise, training runs on the CPU.

## Run

Train either model from the project directory:

```bash
python biagram.py
python gpt.py
```

To save the transformer output:

```bash
python gpt.py > out.txt 2>&1
```

The output includes the model size, periodic training and validation losses, and a generated text sample. The transformer is considerably larger than the bigram model and may take longer to train on a CPU.
