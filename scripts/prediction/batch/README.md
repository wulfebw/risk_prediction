
<!-- MarkdownTOC -->

- [Batch Prediction](#batch-prediction)
    - [Fitting a Prediction Network to a Dataset](#fitting-a-prediction-network-to-a-dataset)

<!-- /MarkdownTOC -->

# Batch Prediction
- this directory contains code used for batch prediction tasks

## Fitting a Prediction Network to a Dataset
- first, you'll need a dataset, see `scripts/collection/` for information on how to generate one
- next, you can fit a network to your dataset using the `fit_predictor.py` script, which will load in a dataset and fit a model to it
- for example: 
```
python fit_predictor.py --dataset_filepath '../../../data/datasets/risk.h5'
```
- there are a large number of settings for this, please see `prediction_flags.py` for more information on those
- one notable point is that you can do either regression or classification prediction, which is controlled by setting the `task_type` and `num_target_bins` flags