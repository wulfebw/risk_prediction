
<!-- MarkdownTOC -->

- [Dataset Collection](#dataset-collection)
    - [Collection](#collection)

<!-- /MarkdownTOC -->

# Dataset Collection
- this directory contains scripts for generating datasets
- there are a variety of datasets that can be generated
- 'heuristic' datasets are those with something heuristic about them - for example, they might rely on simple rules for how they initialize scenes
- datasets that are not heurisitc are typically generated using a method that learns from data how to e.g., initialize the scene or act within that scene

## Collection
- to collect a basic dataset run:
```
julia run_collect_dataset.jl --num_scenarios 1 --num_monte_carlo_runs 1
```
- This will generate a dataset from a single scenario (i.e., a roadway, scene, behavior model tuple), output some information about that dataset, and save it to a file in the data directory.