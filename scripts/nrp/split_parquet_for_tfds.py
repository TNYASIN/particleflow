"""
Split a single large parquet file (many events per row) into many
single-row parquet files, for use as the manual_dir input to build_tfds.py.

split_sample() in mlpf/heptfds/muoncollider_pf/utils_edm.py divides data
across BUILDER_CONFIGS and train/test by FILE COUNT, not event count.
This means:
  - A single input file will crash split_sample() (train/test split
    produces an empty list on one side).
  - Too few files relative to num_splits (10) will silently starve some
    configs of train or test data — you need meaningfully more files
    than num_splits, ideally hundreds+.

Usage:
    python split_parquet_for_tfds.py <input.parquet> <output_dir> [--events-per-file N]
"""
import pyarrow.parquet as pq
import awkward as ak
import os

infile = '/Users/tanya/Downloads/ttbar_reco.parquet'
outdir = '/Users/tanya/Downloads/ttbar_reco-v06-merged'
rows_per_file = 10   # merges 10 original rows (10*10=100 events) into 1 output row/file
os.makedirs(outdir, exist_ok=True)

print('Loading', infile)
table = pq.read_table(infile)
table = table.replace_schema_metadata({})
arr = ak.from_arrow(table)
n_rows = len(arr)
print('Total original rows:', n_rows)

fields = ak.fields(arr)
n_files = (n_rows + rows_per_file - 1) // rows_per_file

for i in range(n_files):
    start, end = i * rows_per_file, min((i + 1) * rows_per_file, n_rows)
    merged = {}
    for f in fields:
        sub = arr[f][start:end]          # shape: (chunk_size, 10, ...)
        merged[f] = ak.flatten(sub, axis=1)  # shape: (chunk_size*10, ...)
    rec = ak.Record(merged)
    outpath = os.path.join(outdir, f'merged_{i:05d}.parquet')
    ak.to_parquet(rec, outpath)
    if i % 100 == 0:
        print(f'Wrote {outpath}')
print('Done. Wrote', n_files, 'files.')
