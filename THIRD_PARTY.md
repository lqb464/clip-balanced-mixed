# Third-party source

- `rde/`: copied from the user's completed RDE run under
  `results_training_old/tbps/rde/2024-CVPR-RDE`, originally from
  [RDE](https://github.com/XLearning-SCU/2024-CVPR-RDE) and the
  [TBPS repository](https://github.com/HoangVo-Prog/tbps).
  The original RDE MIT license is retained at `rde/LICENSE`.
- RDE's CLIP implementation derives from
  [OpenAI CLIP](https://github.com/openai/CLIP), copyright 2021 OpenAI, MIT.
  The source retains its attribution. The tokenizer's BPE vocabulary is
  bundled at `rde/data/bpe_simple_vocab_16e6.txt.gz`.
- `sampler.py`: Balanced Mixed implementation from the user's
  `tbps-sampler` repository, MIT license retained at the repository root.
- Facebook's checkpointer attribution is retained in
  `rde/utils/checkpoint.py`.

Model, losses, TSE heads, dataset augmentation/tokenization and optimizer/scheduler
are unchanged from the historical run. Integration changes are confined to options,
the Balanced Mixed dataloader branch, checkpoint/resume, metrics export and launch scripts.
No training data or user checkpoints are redistributed.
