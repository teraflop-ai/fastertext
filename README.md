# Installion
```
uv pip install fastertext
```

# Usage
On a model such as glotlid you should expect around a 7x-10x speedup compared to regular fasttext.
```bash
! wget https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v3.bin
```
```py
from fastertext import load_model

model = load_model("model_v3.bin")
labels, probs = model.batch(["hello world", "bonjour le monde"], k=2)

id2label = model.get_labels()
print(id2label[int(labels[0, 0])], probs[0, 0])
```

# Benchmark on longer Wikipedia documents
```
fastertext: 2.93s  34,114 texts/s
fasttext:   17.84s  5,604 texts/s
speedup:    6.1x
top-1 agreement: 100.0000%
```

# Citations
```bibtex
@misc{shippole2026fastertext,
  author       = {Shippole, Enrico},
  title        = {fastertext: embarrassingly parallel fasttext batch inference in rust},
  year         = {2026},
  howpublished = {\url{https://github.com/teraflop-ai/fastertext}}
}

@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}

@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}

@misc{messense_fasttext_rs,
  author       = {messense},
  title        = {fasttext-rs: fastText Rust binding},
  howpublished = {\url{https://github.com/messense/fasttext-rs}}
}

@misc{wang_fasttext_parallel,
  author       = {Congyu Wang},
  title        = {fasttext-parallel: multithreaded batch processing for fasttext},
  howpublished = {\url{https://github.com/Congyuwang/fasttext-parallel}}
}
```