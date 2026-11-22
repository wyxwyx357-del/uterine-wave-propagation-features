# Publication validation

- Content/schema inventory:28,553 files inspected,6,243 content-positive matches,0 inspection errors. Identity-bearing paths and the full mapping remain local.
- Existing code relocated:44 analysis modules,4 historical assembly scripts,22 byte-identical measurement modules. Import/resource addressing changes preserve numerical constants. Original active sources/results remain unchanged.
- Existing aggregate result tables:25 copied with source path/hash provenance. No patient OOF, master, identity map, workbook, video, annotation, tracking array or fitted patient model is published.
- Synthetic suite:21 tests passed,0 failures/errors. It includes actual archived Ridge inner selection/outer refit on synthetic inputs, analytical saved tracks→A→ER→record MSE, segment/gap boundaries, decomposition and paired foldwise evaluation.
- Separate fixed Tiny MLP synthetic verification passed on CPU PyTorch2.6.0: reproducible seeds, independent NumPy logits, train-only preprocessing, validation-BCE early stopping and no test-based epoch selection.
- Exact private-source name/identifier checks and Python phone/national-ID literal checks found0 matches. No literal local drive paths or patient-level result schemas occur in the added analysis files.
- Public data alone cannot reproduce historical patient metrics: canonical private master/fold-job hashes are provided, while their contents remain private. No patient reanalysis is claimed or performed.
- Original README bytes are preserved, with an analysis-index link appended. The existing numerical-tests CI now installs its previously missing pandas/scientific and headless-OpenCV test dependencies; the OpenCV version constraint preserves the frozen NumPy1.26 environment. No pre-existing tracking, feature, threshold or model implementation is overwritten.
- Git diff review preserves original CRLF and source bytes. The only whitespace-check notices are pre-existing terminal blank lines in4 byte-identical measurement source files: `__init__.py`, `_lk.py`, `geometry.py`, `tracking.py`. They are deliberately not reformatted; there are no conflict markers or unrelated modifications.
- Primary remains Clinical9 Ridge versus Clinical9+MSE Ridge. Tiny MLP, alternate baselines, other dynamic and static negative/inconclusive results retain their stated sensitivity/exploratory roles. Model search remains closed.
