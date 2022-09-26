# Active Learning for SAT-Solver Benchmarking

This repository contains the paper and code of *Active Learning for SAT-Solver Benchmarking*.
All data and immediate results are available in a [separate repository](https://github.com/mathefuchs/al-for-sat-solver-benchmarking-data).

## Repository Structure

* The folder `paper` contains the LaTeX source of the paper.
* The folder `plots` contains the pgf-plots used within the publication.
* The folder `src` contains the code of the main experiments.
* The notebook `anni_results.ipynb` analyzes the results produced by the code under `src`.
* The notebook `prepare_data.ipynb` shows how we prepare the underlying data for our experiments.
* `README.md` is this document.
* `run_debug.sh` starts the main experiment on a single core with debug-logging turned on.
* `run.sh` starts the main experiment on all available cores.

## Setup

* Clone this repository.
* Clone the [data repository](https://github.com/mathefuchs/al-for-sat-solver-benchmarking-data) in the same parent folder as this repository and init submodules with `git submodule update --init --recursive` within the data repository's directory.
* Change the working directory back to this repository and create a `conda` environment with the following commands

``` sh
conda env create -f environment.yml
conda activate al_env  
```

* Run the final experiment with the best configurations using `sh ./run.sh` (runs the experiment on all available cores) or `sh ./run_debug.sh` (runs the experiment on a single core with debug-logging turned on).
* Evaluate the results with the code in the `anni_results.ipynb` Jupyter-Notebook.
