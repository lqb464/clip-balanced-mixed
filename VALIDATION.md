# Validation

Seven CPU unit tests passed on Python 3.12, PyTorch 2.6.0+cpu and torchvision 0.21.0+cpu on Windows.

- Full-batch SDM updates match gradient-cache updates, including an uneven final microbatch.
- A small instance of the actual CLIP transformer implementation matches full-batch gradient replay.
- Restored AdamW state produces the same next parameter update.
- Dataset captions retain their original image pairing; row/epoch augmentation reproduces after resume.
- Global negative caches exclude same-PID candidates and map text-to-image ranks to unique image representatives.
- Sampler coverage includes each row exactly once and regenerates the same remaining order after resume.
- Retrieval uses a unique-image gallery and gives the expected perfect and reversed-ranking scores.

The full 60-epoch experiment and CUDA AMP on T4 have not been executed here. The notebook runs the unit tests before creating the cache and launching training. The first training batch logs peak allocated GPU memory. No R@1 improvement is claimed before both runs finish.
