# SegTHOR exploration

Open [segthor_exploration.ipynb](segthor_exploration.ipynb) in VS Code. Select the
project's `ai4mi/bin/python` kernel and choose **Run All**. Saved outputs can be
read immediately; interactive controls require a running kernel.

The notebook uses inline plots and widgets, so X11 forwarding and Tkinter are
unnecessary. It needs no GPU. The original-volume audit reads each patient once
and can take several minutes. Set `RUN_RAW_AUDIT = False` in the configuration
cell to skip it when focusing on processed slices or training results.

For a fresh environment, run this from the repository root:

```bash
source ai4mi/bin/activate
python -m pip install -r requirements-notebooks.txt
```

The configuration cell controls dataset, original-volume, training-run, and
export paths. The notebook reads the data and model predictions, and writes CSV
tables and PNG figures under `results/segthor/data_exploration/`. Rerun it after
changing the data or training results to refresh the snapshot.
