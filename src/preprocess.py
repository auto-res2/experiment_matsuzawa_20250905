[UPDATED]
```python
# ... previous imports & constants ...

FIG_DIR = ROOT / ".research" / "iteration5" / "images"  # updated per instructions
for d in (DATA_DIR, OUTPUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
```
