from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from utils_edm import (
    NUM_SPLITS,
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    generate_examples,
    split_sample,
)

_DESCRIPTION = """
Muon Collider dataset with mu+mu- -> ttbar.
  - X: reconstructed tracks and clusters, variable number N per event
  - ytarget: stable generator particles, zero-padded to N per event
  - ycand: baseline particle flow candidates, zero-padded to N per event
"""

_CITATION = """
FIXME
"""


class MuoncolliderTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.2.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release",
        "1.1.0": "change in target defn",
        "1.2.0": "100k events ",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place the muon collider parquet files in the manual_dir (e.g. /data/dataset).
    """

    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(MuoncolliderTtbarPf, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(
                        shape=(None, max(len(X_FEATURES_TRK), len(X_FEATURES_CL))),
                        dtype=np.float32,
                    ),
                    "ytarget":    tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "ycand":      tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "genmet":     tfds.features.Scalar(dtype=np.float32),
                    "genjets":    tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK,
                x_features_cluster=X_FEATURES_CL,
                y_features=Y_FEATURES,
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # parquets live directly in manual_dir (no subdirectory like CLD)
        path = dl_manager.manual_dir
        return split_sample(Path(path), self.builder_config, num_splits=NUM_SPLITS)

    def _generate_examples(self, files):
        return generate_examples(files)
