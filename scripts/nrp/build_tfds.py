import os
import sys

data_dir = sys.argv[1]      # e.g. /data/users/tanya/tfds
manual_dirs = sys.argv[2:]  # one or more parquet dirs

# Pass extra dirs (beyond the first) via env so split_sample can glob them
if len(manual_dirs) > 1:
    os.environ["EXTRA_PARQUET_DIRS"] = ":".join(manual_dirs[1:])

sys.path.insert(0, "/data/users/tanya/particleflow/mlpf/heptfds/muoncollider_pf")
import ttbar  # noqa: E402 — registers MuoncolliderTtbarPf with TFDS

import tensorflow_datasets as tfds

for config in ttbar.MuoncolliderTtbarPf.BUILDER_CONFIGS:
    print("Building config:", config.name)
    builder = ttbar.MuoncolliderTtbarPf(config=config, data_dir=data_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=manual_dirs[0])
    )
    print("Done:", config.name)

print("All configs built.")
