
<!-- MarkdownTOC -->

- [Extracting a Dataset from NGSIM](#extracting-a-dataset-from-ngsim)
    - [Extraction Setting](#extraction-setting)

<!-- /MarkdownTOC -->

# Extracting a Dataset from NGSIM
- deriving a dataset from NGSIM entails selecting segments of the trajectories for each vehicle, extracting features for the initial part of that trajectory, and extracting targets for the remaining part 
- for how to download and preprocess NGSIM data, see the [NGSIM.jl](https://github.com/sisl/NGSIM.jl) package

## Extraction Setting
- the file `extract_ngsim_dataset.jl` contains hardcoded properties of the dataset
- this includes for example the amount of time the target values are computed over and the number of feature timesteps collected
- change those values as described in the file 
- then run `julia extract_ngsim_dataset.jl` to extract the dataset and save it to a file
- the script can run in parallel for each trajectory dataset in NGSIM, of which there are 6, so if you have 6 cores available, run `julia -p 6 extract_ngsim_dataset.jl` instead
