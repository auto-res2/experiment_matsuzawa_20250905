[UPDATED]
1. Fixed OrthogonalProjector initialization – replaced deprecated Tensor.orthogonal_() call with nn.init.orthogonal_.
2. LatentReplayBuffer.sample now stacks labels with torch.stack to avoid dtype/shape issues.
