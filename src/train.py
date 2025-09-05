[UPDATED]
```python
    def mine(self, dataset, limit: int = 2_000) -> Tuple[List[int], faiss.Kmeans]:  # type: ignore[valid-type]
        """Return cluster-id per sampled element & the fitted k-means object."""
        import random

        # Prepare feature matrix ------------------------------------------- #
        heatmaps: List[torch.Tensor] = []
        sample_idxs = random.sample(range(len(dataset)), k=min(limit, len(dataset)))
        for idx in sample_idxs:
            # WILDS datasets return (x, y, metadata). We only need the image.
            img, *_ = dataset[idx]  # grab the first element and ignore the rest
            img = img.unsqueeze(0).to(dtype=DTYPE, device=DEVICE)
            heatmaps.append(self._heatmap(img).cpu())
        maps = torch.stack(heatmaps).view(len(heatmaps), -1).numpy().astype("float32")
        faiss.normalize_L2(maps)

        # Decide whether FAISS has GPU support ----------------------------- #
        has_gpu = hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available()
        km = faiss.Kmeans(d=maps.shape[1], k=self.k, niter=20, gpu=has_gpu, verbose=True)
        km.train(maps)
        _, I = km.index.search(maps, 1)  # noqa: N806 – FAISS style
        return I.squeeze().tolist(), km
```

```python
    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        correct = total = 0
        for x, y, *_ in self.val_loader:  # wilds returns (x,y,metadata)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = self.model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        self.model.train()
        return correct / total if total > 0 else 0.0
```
