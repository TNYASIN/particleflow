import awkward as ak
import numpy as np

NUM_SPLITS = 10

X_FEATURES_TRK = [
    "elemtype", "pt", "eta", "sin_phi", "cos_phi", "p",
    "chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit",
    "tanLambda", "D0", "omega", "Z0", "time",
]
X_FEATURES_CL = [
    "elemtype", "et", "eta", "sin_phi", "cos_phi", "energy",
    "position.x", "position.y", "position.z", "iTheta",
    "energy_ecal", "energy_hcal", "energy_other", "num_hits",
    "sigma_x", "sigma_y", "sigma_z",
]
Y_FEATURES = [
    "PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy",
    "ispu", "generatorStatus", "simulatorStatus",
    "gp_to_track", "gp_to_cluster", "jet_idx", "particle_number",
]
labels = [0, 211, 130, 22, 11, 13]

N_X_FEATURES = max(len(X_FEATURES_CL), len(X_FEATURES_TRK))
N_Y_FEATURES = len(Y_FEATURES)


def split_list(lst, x):
    if not lst:
        return [[] for _ in range(x)]
    sublist_size = max(1, len(lst) // x)
    result = [lst[i * sublist_size: (i + 1) * sublist_size] for i in range(x - 1)]
    result.append(lst[(x - 1) * sublist_size:])
    while len(result) < x:
        result.append([])
    return result


def split_sample(path, builder_config, num_splits=NUM_SPLITS, test_frac=0.9):
    import os
    from pathlib import Path as _Path
    files = sorted(list(path.glob("*.parquet")))
    extra_dirs = os.environ.get("EXTRA_PARQUET_DIRS", "")
    for extra in extra_dirs.split(":"):
        if extra:
            extra_path = _Path(extra)
            files += sorted(list(extra_path.glob("*.parquet")))
    files = sorted(files)
    print("Found {} total files across all data dirs".format(len(files)))
    assert len(files) > 0

    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test  = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0

    split_index = int(builder_config.name) - 1
    files_train_split = split_list(files_train, num_splits)
    files_test_split  = split_list(files_test, num_splits)

    return {
        "train": generate_examples(files_train_split[split_index]),
        "test":  generate_examples(files_test_split[split_index]),
    }


def _arr(ak_obj, dtype=np.float32):
    return np.array(ak.to_list(ak_obj), dtype=dtype)


def _prepare_event(ret, iev, has_ycand):
    """Extract one event (index iev) from an awkward parquet record.

    ak.from_parquet on a single-row parquet returns a Record, so
    ret["X_track"] is already the event array — no outer [0] needed.
    """
    X1 = _arr(ret["X_track"][iev]).reshape(-1, 16)    # (n_tracks, 16)
    X2 = _arr(ret["X_cluster"][iev]).reshape(-1, 17)  # (n_clusters, 17)

    if X1.shape[0] == 0:
        X1 = np.zeros((0, N_X_FEATURES), dtype=np.float32)
    if X2.shape[0] == 0:
        X2 = np.zeros((0, N_X_FEATURES), dtype=np.float32)
    if X1.shape[1] < N_X_FEATURES:
        X1 = np.pad(X1, [[0, 0], [0, N_X_FEATURES - X1.shape[1]]])

    ytarget_track   = _arr(ret["ytarget_track"][iev]).reshape(-1, N_Y_FEATURES)
    ytarget_cluster = _arr(ret["ytarget_cluster"][iev]).reshape(-1, N_Y_FEATURES)

    if has_ycand:
        ycand_track   = _arr(ret["ycand_track"][iev]).reshape(-1, N_Y_FEATURES)
        ycand_cluster = _arr(ret["ycand_cluster"][iev]).reshape(-1, N_Y_FEATURES)
    else:
        ycand_track   = np.zeros_like(ytarget_track)
        ycand_cluster = np.zeros_like(ytarget_cluster)

    genmet    = float(ret["genmet"][iev])
    genjet    = _arr(ret["genjet"][iev])
    targetjet = _arr(ret["targetjet"][iev])

    if genjet.ndim < 2 or genjet.shape[0] == 0:
        genjet = np.zeros((0, 4), dtype=np.float32)
    if targetjet.ndim < 2 or targetjet.shape[0] == 0:
        targetjet = np.zeros((0, 4), dtype=np.float32)

    X       = np.concatenate([X1, X2])
    ytarget = np.concatenate([ytarget_track, ytarget_cluster])
    ycand   = np.concatenate([ycand_track, ycand_cluster])

    if ytarget.shape[0] != X.shape[0] or ycand.shape[0] != X.shape[0]:
        raise ValueError("Shape mismatch: X={} ytarget={} ycand={}".format(X.shape, ytarget.shape, ycand.shape))

    ytarget[:, 0] = [labels.index(int(p)) if int(p) in labels else 0 for p in ytarget[:, 0]]
    ycand[:, 0]   = [labels.index(int(p)) if int(p) in labels else 0 for p in ycand[:, 0]]

    return X, ytarget, ycand, genmet, genjet, targetjet


def prepare_data(fn):
    """Load all events from a parquet file. Each file may contain 1 or more events."""
    ret = ak.from_parquet(fn)
    n_events = len(ret["X_track"])   # Record: top-level is the event array
    has_ycand = "ycand_track" in ak.fields(ret)
    results = []
    for iev in range(n_events):
        results.append(_prepare_event(ret, iev, has_ycand))
    return results


def generate_examples(files):
    example_idx = 0
    for fi in files:
        try:
            events = prepare_data(fi)
        except Exception as e:
            print("Skipping {}: {}".format(fi, e))
            continue
        for X, ytarget, ycand, genmet, genjet, targetjet in events:
            yield str(example_idx), {
                "X":          X.astype(np.float32),
                "ytarget":    ytarget.astype(np.float32),
                "ycand":      ycand.astype(np.float32),
                "genmet":     float(genmet),
                "genjets":    genjet.astype(np.float32),
                "targetjets": targetjet.astype(np.float32),
            }
            example_idx += 1
