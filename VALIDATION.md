# Validation

The experiment retains the historical RDE model, loss functions, BGE/TSE heads,
dataset augmentation/tokenizer and solver. Source hashes are checked against the
files accompanying the completed old run, not a reimplementation of the paper.

CPU checks:

1. Original core-source SHA-256 matches (normalizing CRLF only).
2. Default options match the historical configs.yaml, including batch64,
   Identity K=4 reference, both augmentations, test selection and 60 epochs.
3. Actual RDE DataLoader accepts Balanced Mixed: correct batch/token/image shapes,
   full source coverage, PID diversity and enabled mining.
4. Cache validation rejects different weights, and negative ranking excludes
   same PID while deduplicating text-to-image gallery images.
5. Original Adam parameter groups, bias decay and TSE-specific LR are preserved;
   scheduler, optimizer, model and RNG are serializable/restorable.
6. Original RDE retrieval metric returns the expected values on a known ranking.
7. Comparison reads evaluated best/last checkpoints separately; rejects wrong configs.
8. Launcher keeps batch64, 60 epochs and K=4, and detects epoch-boundary resume files.
9. Original RDE forward/backward runs with a tiny CLIP backbone and both TSE heads.
   In this CPU-only test, the original TAL .cuda() transfer is mocked to CPU;
   no mathematical code is modified.
10. Training-loop pause/resume smoke test preserves best state, sampler phase
    seeds and epoch history without repeating the completed epoch. Toy model and
    mocked GMM labels/device transfers are used only in this test.
11. The original get_loss/GMM function fits two mixtures and returns correctly
    indexed binary labels on a known synthetic low/high-loss dataset.

The notebook generator checks Python syntax for all five code cells.

Limitations: no full GPU training, CUDA-memory measurement or 60-epoch result is
claimed from these checks. Historical comparison is one seed and selects best
checkpoints on the test split, as the old run did. Equal epochs do not imply
equal training samples/steps when sampler coverage differs.
