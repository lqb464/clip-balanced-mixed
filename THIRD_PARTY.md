# Attribution

- `sampler.py`: Balanced Mixed from the user's local `tbps-sampler/irra/datasets/sampler_mining.py`, copied without changing its sampling algorithm.
- `vendor/clip_model.py`: IRRA's CLIP adaptation, based on OpenAI CLIP. Changes in this repo make loading errors fail explicitly and import the download module explicitly.
- `vendor/simple_tokenizer.py` and `data/bpe_simple_vocab_16e6.txt.gz`: CLIP tokenizer and vocabulary copied from the same source repository.
- SDM follows `irra/model/objectives.py::compute_sdm`; the new trainer uses a fixed scale of 50 and PID-normalized targets.
- Source repository: https://github.com/HoangVo-Prog/tbps-sampler
- IRRA: https://github.com/anosorae/IRRA
- OpenAI CLIP: https://github.com/openai/CLIP

The source repository's MIT license is preserved in `LICENSE`.

## OpenAI CLIP license

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
