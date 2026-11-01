# Later captures preserved during artifact review

These files preserve the exact working-tree bytes found after the Trial 2 push. `capture01` is an archival label assigned during review, not a verified execution-attempt number. The generating chat/process and execution time are unknown; the files may have come from reruns or delayed commands. No experiment was rerun during this archival work.

Tracked logs at their original paths were restored byte-for-byte from the revision recorded in `manifest.json`. Previously untracked logs were moved here. The manifest records original paths and SHA-256 hashes of these preserved captures.

Original `metrics.csv` files, source, tests, prompts, and responses were left unchanged. These later captures are supplementary evidence and do not replace the original measured results. No new metrics were inferred from partial logs.

The later race capture reports real 0.05 seconds. The original capture and metrics retain real 0.09 seconds. Additional preserved files contain demonstration outputs; the original committed flow capture was empty.
