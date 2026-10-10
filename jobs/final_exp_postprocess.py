#!/usr/bin/env python3
"""Matched postprocessing evaluations for the selected M1 model."""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import eval as evaluation
import postprocessing as pp

RUN = ROOT / "results/segthor_full_seed43/final_exp_M1_A2_unet_large"
DATA = ROOT / "data/segthor_full_seed43/final_exp_A2_A1_crop"
SOURCE = ROOT / "data/segthor_train_full"
METHODS = {
    "none": None,
    "largest_connected_components": partial(pp.keep_largest_connected_components, k=1, classes=[2, 3, 4], connectivity=26),
    "fill_holes": partial(pp.fill_holes, classes=[2], connectivity=6),
    "opening": partial(pp.opening, classes=[2], iterations=1, connectivity=6),
    "closing": partial(pp.closing, classes=[2], iterations=1, connectivity=6),
    "salt_and_pepper": partial(pp.salt_and_pepper, classes=[2], kernel_size=3),
    "foreground_components": partial(pp.keep_foreground_components, k=4),
    "small_components": partial(pp.remove_small_components, min_size=100),
    "anatomy_aware_filtering": None,
    "dense_crf": None,
}
COMBINATIONS = {
    "anatomy_safe": ["anatomy_safe"],
    "anatomy_safe_fill": ["anatomy_safe", "fill_holes"],
    "crf_aorta": ["crf_aorta"],
    "crf_aorta_anatomy_fill": ["crf_aorta", "anatomy_safe", "fill_holes"],
}
METHODS.update(dict.fromkeys(COMBINATIONS))
METRICS = ["dice", "hausdorff_distance_95", "average_surface_distance"]


def check_inputs():
    expected = {p.stem for p in (DATA / "val/geometry").glob("*.json")}
    actual = {p.name[:-7] for p in (RUN / "pred_volumes").glob("*.nii.gz")}
    if len(expected) != 10 or actual != expected:
        raise ValueError("Expected the same ten A2 validation patients and M1 predictions")
    for pid in sorted(expected):
        pred = nib.load(RUN / "pred_volumes" / f"{pid}.nii.gz")
        for path in [SOURCE / "train" / pid / "GT.nii.gz", SOURCE / "train" / pid / f"{pid}.nii.gz"]:
            image = nib.load(path)
            if image.shape != pred.shape or not np.allclose(image.affine, pred.affine, atol=1e-4, rtol=0):
                raise ValueError(f"Native-grid mismatch: {pid}")
    return sorted(expected)


def score(pred_folder, output, processor):
    rows, matrices = evaluation.evaluate_dataset(
        pred_folder, str(SOURCE / "train/{id_}/GT.nii.gz"),
        num_classes=5, metrics=METRICS, postprocess=processor,
        save_folder=output / "pred_volumes", confusion_classes=5)
    dest = output / "eval_metrics.csv"
    evaluation.save_csv(rows, dest)
    summary = evaluation.summarize(rows, METRICS)
    evaluation.save_csv(summary, output / "eval_metrics_summary.csv")
    evaluation.save_confusion_matrices(matrices, evaluation.SEGTHOR_CLASS_NAMES, dest)
    evaluation.print_summary(summary, METRICS)



def transfer_crf_aorta(patients, output):
    source = RUN / "postprocessing/dense_crf/pred_volumes"
    if {p.name[:-7] for p in source.glob("*.nii.gz")} != set(patients):
        raise ValueError("Missing or mismatched saved Q7 patients")
    dest = output / "aorta_transfer"
    dest.mkdir(exist_ok=True)
    for pid in patients:
        raw = nib.load(RUN / "pred_volumes" / f"{pid}.nii.gz")
        crf = nib.load(source / f"{pid}.nii.gz")
        if crf.shape != raw.shape or not np.allclose(crf.affine, raw.affine, atol=1e-4, rtol=0):
            raise ValueError(f"CRF-grid mismatch: {pid}")
        labels = np.asarray(raw.dataobj).copy()
        candidate = np.asarray(crf.dataobj)
        if not np.isin(candidate, [0, 1, 2, 3, 4]).all():
            raise ValueError(f"Invalid CRF labels: {pid}")
        editable = (labels == 0) | (labels == 4)
        labels[editable] = np.where(candidate[editable] == 4, 4, 0)
        nib.save(nib.Nifti1Image(labels, raw.affine, raw.header.copy()), dest / f"{pid}.nii.gz")
    return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    patients = check_inputs()
    output = RUN / "postprocessing" / args.method
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "postprocessing_config.json"
    if args.prepare_only or not config_path.exists():
        config = json.loads((ROOT / "postprocessing_config.json").read_text())
        if args.method in COMBINATIONS:
            steps = COMBINATIONS[args.method]
            if "anatomy_safe" in steps:
                saved = RUN / "postprocessing/anatomy_aware_filtering/postprocessing_config.json"
                config["anatomy_aware_filtering"] = json.loads(saved.read_text())["anatomy_aware_filtering"]
                config["anatomy_aware_filtering"]["classes"].pop("3", None)
            if "crf_aorta" in steps:
                saved = RUN / "postprocessing/dense_crf/postprocessing_config.json"
                config["dense_crf"] = json.loads(saved.read_text())["dense_crf"]
            config["combination"] = {
                "steps": steps,
                "raw_predictions": str(RUN / "pred_volumes"),
                "crf_predictions": str(RUN / "postprocessing/dense_crf/pred_volumes") if "crf_aorta" in steps else None,
                "crf_editable_labels": [0, 4] if "crf_aorta" in steps else [],
            }
        config["dense_crf"]["image_pattern"] = str(SOURCE / "train/{id_}/{id_}.nii.gz")
        config["dense_crf"]["probability_pattern"] = str(RUN / "postprocessing/dense_crf/probabilities/{id_}.nii.gz")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    config = json.loads(config_path.read_text())
    if args.prepare_only:
        return
    processor = METHODS[args.method]
    pred_folder = RUN / "pred_volumes"
    if args.method in COMBINATIONS:
        steps = COMBINATIONS[args.method]
        if "crf_aorta" in steps:
            pred_folder = transfer_crf_aorta(patients, output)
        processors = []
        if "anatomy_safe" in steps:
            processors.append((partial(pp.anatomy_aware_filtering, config=config["anatomy_aware_filtering"]), ("spacing",)))
        if "fill_holes" in steps:
            processors.append((partial(pp.fill_holes, classes=[2], connectivity=6), ()))
        processor = evaluation.PostprocessingPipeline(processors) if processors else None
    if args.method == "anatomy_aware_filtering":
        processor = evaluation.PostprocessingPipeline([
            (partial(pp.anatomy_aware_filtering, config=config[args.method]), ("spacing",))])
    if args.method == "dense_crf":
        settings = config["dense_crf"]
        argmax_dir = output / "probability_argmax"
        argmax_dir.mkdir(exist_ok=True)
        for pid in patients:
            image = nib.load(SOURCE / "train" / pid / f"{pid}.nii.gz")
            prob = nib.load(settings["probability_pattern"].format(id_=pid))
            if prob.shape != image.shape + (5,) or not np.allclose(prob.affine, image.affine, atol=1e-4, rtol=0):
                raise ValueError(f"Probability-grid mismatch: {pid}")
            probabilities = np.asarray(prob.dataobj)
            for z in range(image.shape[2]):
                slab = probabilities[:, :, z, :]
                if not np.isfinite(slab).all() or slab.min() < 0 or slab.max() > 1 or not np.allclose(slab.sum(-1), 1, atol=1e-5):
                    raise ValueError(f"Invalid probabilities: {pid}, slice {z}")
            labels = probabilities.argmax(-1).astype(np.uint8)
            header = image.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(labels, image.affine, header), argmax_dir / f"{pid}.nii.gz")
            del probabilities, labels
        score(argmax_dir, output / "argmax_control", None)
        pred_folder = argmax_dir
        processor = evaluation.PostprocessingPipeline(
            [(partial(pp.dense_crf, config=settings), ("probabilities", "image", "spacing"))],
            probability_pattern=settings["probability_pattern"], image_pattern=settings["image_pattern"])
    score(pred_folder, output, processor)


if __name__ == "__main__":
    main()
