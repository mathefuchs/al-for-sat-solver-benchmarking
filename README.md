# Active Learning for SAT Solver Benchmarking

This repository contains the code and text of the paper

> Fuchs, Bach, Iser. "Active Learning for SAT Solver Benchmarking"

accepted at the conference [TACAS 2023](https://etaps.org/2023/tacas).
You can find the corresponding complete experimental data (inputs as well as results) within [this data repository](https://github.com/mathefuchs/al-for-sat-solver-benchmarking-data).

This document provides:

* an outline of the repository structure
* steps to reproduce the experiments, including setting up a virtual environment
* guidelines for developers who want to modify or extend the code base

## Repository Structure

* The folder `paper` contains the LaTeX source of the paper.
* The folder `plots` contains the pgf-plots used within the publication. All plots therein are generated with the `anni_results.ipynb` Jupyter notebook.
* The folder `src` contains the code of the experiments.
* Additionally, there are the following files in the root directory:
  * The notebook `anni_results.ipynb` analyzes the results produced by the code under `src` and generates all plots within `plots`.
  * `LICENSE` describes the repository's licensing.
  * The Jupyter notebook `prepare_data.ipynb` shows how we prepare the underlying data for our experiments.
  * `README.md` is this document.
  * `requirements.txt` is a list of all required `pip` packages for reproducibility.
  * `run_debug.(sh|ps1)` starts the main experiment on a single core with debug logging turned on.
  * `run.(sh|ps1)` starts the main experiment on all available cores.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all the necessary dependencies.
Our code is implemented in Python (version 3.9.12; other versions, including lower ones, might also work).

We used `virtualenv` (version 20.17.1; other versions might also work) to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Let's assume the Python executable is located at <path/to/python>.
Next, you install `virtualenv` with

| Linux + Macos                               | Windows (powershell)                        |
|---------------------------------------------|---------------------------------------------|
| `python -m pip install virtualenv==20.17.1` | `python -m pip install virtualenv==20.17.1` |

To create a virtual environment for this project, you have to clone this repository first.
Thereafter, change the working directory to this repository's root folder.
Run the following commands to create the virtual environment and install all necessary dependencies.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
<td>

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
</tr>
</table>

## Reproducing the Experiments

Make sure that, both, this repository and the [data repository](https://github.com/mathefuchs/al-for-sat-solver-benchmarking-data) are cloned within the same root folder.
Also, create a virtual environment as described above.
The script `run.(sh|ps1)` runs the experiment on all available cores.
The script `run_debug.(sh|ps1)` runs the experiment on a single core with debug logging turned on.
Note that both scripts skip results that are already present.
To re-generate all results, delete all files within the folder `al-for-sat-solver-benchmarking-data/pickled-data/end_to_end`.
Do not delete the folder itself.
Then, you can simply run the final experiment with:

| Linux + MacOS                    | Windows (powershell)             |
|----------------------------------|----------------------------------|
| `sh run.sh` or `sh run_debug.sh` | `.\run.ps1` or `.\run_debug.ps1` |

To obtain plots from the data, use the Jupyter notebook `anni_results.ipynb`.
Running the final experiment takes about a week on a system with 32 cores (Setup: `AMD EPYC 7551 32-Core Processor`).

## Tweaking and Extending the Code

A good starting point are the files [final_experiments.py](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/final_experiments.py) and [experiment.py](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/experiment.py) if you want to modify or extend the code.
The former lists all experiments that are executed within the experimental run triggered by the script `run.(sh|ps1)`.
The latter defines the building blocks needed for an experiment.
To add your experiment, create a new instance of [Experiment](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/experiment.py#L467) with the factory method [Experiment.custom()](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/experiment.py#L467).
The necessary parameters are:

* `instance_filter`: The name of the dataset. Must be a file within `al-for-sat-solver-benchmarking-data/pickled-data/` and of similar structure as `al-for-sat-solver-benchmarking-data/pickled-data/anni_final_df.pkl`.
* `only_hashes`: Set this to `false` as this is only for advanced use-cases.
* `repetitions`: Specify how often you want to repeat the same experiment. Useful if some component of the experiment includes randomness.
* `selection`: The selection strategy. Must be a subclass of [SelectionFunctor](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/selection.py#L10). Refer to other, already implemented, subclasses to get started, e.g., [RandomSampling](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/selection.py#L45).
* `stopping`: The stopping criterion. Must be a subclass of [StoppingPredicate](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/stopping.py#L10). Refer to other, already implemented, subclasses to get started, e.g., [FixedSubsetSize](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/stopping.py#L46).
* `ranking`: The ranking strategy. Must be a subclass of [RankingFunctor](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/ranking.py#L10). Refer to other, already implemented, subclasses to get started, e.g., [PartialObservationBasedRanking](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/src/al_experiments/ranking.py#L51).
