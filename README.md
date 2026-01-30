# Work Sharing and Offloading for Efficient Approximate Threshold-based Vector Join

# Content 
## C++ Files
- `ws_join.cpp` - Our vector join implementation (our SimJoin paper implementation and our ideas, multi-threading possible)
- `naive_join.cpp` - Naive nested loop join (multi-threading possible)
- `CMakeLists.txt` - CMake configuration for building binaries

## Python Scripts
- `download_ann_benchmarks.py` - Helper for downloading ANN benchmarks
- `run_experiments.py` - Main experiment launcher that launches multiple runs
- `normalize_fvecs.py` - Normalize fvecs file (MUST for datasets using angular distance)
- `concatenate_fvecs.py` - Concatenate two fvecs files (Used to generate the fvecs file used for our merged index)

## Bash Scripts
- `build_nsg.sh` - Build NSG indexes using efanna_graph and nsg (our slighlty modified version that also classifies queries into in-distribution and out-of-distribution)
- `run_experiments.sh` - entry point to run the main experiments (`run_experiments.py`)
- `setup_dataset.sh` - script to automatically build nsg graph, normalzize and concatenate vectors

## Docker Configuration
- `build/Dockerfile` - Docker environment with all dependencies


# Setup
```bash
cd build
docker build -t <image_name> .
docker run -dit --privileged -v <path_to_VectorJoin_directory>:/workspace --name <container_name> <image_name>
docker exec -it <image_name> /bin/bash
```

In the docker container:
```bash
cd /workspace
cd build
cmake ..
make -j$(nproc)
```

# Datasets
## Download Datasets

`python3 download_ann_benchmarks.py <dataset_name> --data-dir <data_dir>` 
To list all names of the datasets run: `python3 download_ann_benchmarks.py --list`.

## Rename Datasets
- rename the files `<dataset_dir>/<dataset_name>_train.fvecs` to `<dataset_dir>/<dataset_dir>_base.fvecs`
- rename the files `<dataset_dir>/<dataset_name>_test.fvecs` to `<dataset_dir>/<dataset_dir>_query.fvecs`

Example: mv sift/sift-128-euclidean_train.fvecs sift/sift_base.fvecs

<dataset_dir> should be named the lower case short name of the dataset_name:
- "sift"
- "gist"
- "glove"
- "nytimes"
- "fmnist"
- "coco"
- "laion"
- "imagenet"

## Setup Efanna_graph and Nsg 

- clone [efanna graph](https://github.com/ZJULearning/efanna_graph) and set it up.
- clone [nsg](https://github.com/user-c49lj/nsg) and set it up (this is our slightly modified nsg implementation that also classifies each query into in-distribution and out-of-distribution)

## Build Indices

```bash
run ./setup_dataset.sh <dataset_dir> <0/1 1 If the dataset needs to be normalized>
```

Datasets that need to be normalized are `glove and nytimes`

**Attention:** Only for nytimes dataset change the sixth parameter of `./efanna_graph/tests/test_nndescent` in `build_nsg.sh`from 15 to 50.

# Run experiments

Call `./run_experiments.sh` for some example experiments to run.

Or call `python3 run_experiments.py  --output-dir <output_dir> --dataset <dataset_dir> --experiment --experiment <experiment> --max-concurrent <max_concurrent>` directly.

Parameters:

- `experiment` - can be one of:
    - "naive"
    - "index"
    - "es"
    - "es_hsw"
    - "es_sws"
    - "es_mi"
    - "es_mi_adapt"
    - "all"
- `max_concurrent` - number of workers running concurrently (if a job finishes, the corresponding worker will fetch another job from the queue)
- `output-dir` - location where the results are saved to
- `dataset` - choose the dataset to run experiments on, options are:
    - "sift"
    - "gist"
    - "glove"
    - "nytimes"
    - "fmnist"
    - "coco"
    - "laion"
    - "imagenet"
- `dry-run` - only print the experiment configurations but don't run the actual experiments


# Results Storage

Results are stored in the `<result_dir>/` or `simjoin_results/` directory with subdirectories:
- **`<result_dir>/<subdirectory>/`** - Each experiment run gets its own subdirectory using the specified prefix in the running script, epsilon (distance threshold), # threads, and binary parameters
- **`results/*/join_output.txt`** - Join results and other meta information like processing time, one line per query

The columns in this file are query_id, kappa_i, the number of in reach matches found,  count for number of distance calulations during seedinitaliztion, count for number of distance computation during seedinitaliztion, distance computation count during greedy phase, distance compuation count during bfs phase, query duration spend on each query, max memory usage up to this point
- **`results/*/profile_stats.txt`** - Profiling results
- **`results/*/progress.txt`** - Progress that is output to stdout (you won't see this if you use the launcher) 
- **`results/*/stdout.txt`** - all std::cout command in the script end up here for debugging and logging. Also contains final distance compuation count, total time used and total pairs found.

