[MODIFIED]
@@
-_RESEARCH_ROOT = Path(".research") / "iteration54"
+_RESEARCH_ROOT = Path(".research") / "iteration55"
@@
-            raise RuntimeError(f"Expected Waterbirds split dir missing: {split_dir}")
+            # Some Hub mirrors place the split directories inside an extra
+            # wrapper folder (e.g. `data/waterbird_complete95_forest2water2/`).
+            # If the direct path is missing, fall back to a recursive search
+            # for the first directory whose *name* matches the requested split
+            # (train / val / test).  This keeps the logic robust to minor
+            # layout differences across mirrors while still failing fast if
+            # nothing sensible is found.
+
+            candidates = [p for p in root.glob(f"**/{split}") if p.is_dir()]
+            if not candidates:
+                raise RuntimeError(f"Expected Waterbirds split dir missing: {split_dir}")
+
+            # Pick the shortest path (closest to the root) to avoid accidental
+            # matches inside nested archives or unrelated folders.
+            split_dir = min(candidates, key=lambda p: len(p.parts))
@@