# An organotypic in vitro model of human papillomavirus-associated precancerous lesions allowing automated cell quantification for preclinical drug testing
This repository contains the code used for analyses in the paper _An organotypic in vitro model of human papillomavirus-associated precancerous lesions allowing automated cell quantification for preclinical drug testing_.

To get started right away, skip down to the section
[Getting started](#getting-started).

### Python

Most analyses were performed using Python and can in principle be done on any
standard computer running Windows, macOS or Linux. Most code was run on a
Windows 11 notebook with a 4-core CPU (Intel Core i7), 32GB RAM, without
dedicated GPU. The main Python code for analyses can be found in the [_code_python-results_](code_python-results/) folder.

### Groovy

Code for exporting and handling annotations and detections from QuPath were written in Groovy. The corresponding code can be found in the folder [_code_qupath-groovy_](code_qupath-groovy/).

### Stardist models

The pretrained StarDist model used for nucleus detection in Hematoxylin and eosin stained sections can be found in the folder [_models_stardist_](models_stardist/).

### Data and plots

Intermediate data, final results, and plots can be found in the folders [_data_](data/) and [_plots_](plots/).

## Getting started

To make sure the analysis steps are reproduced as close as possible, I recommend
following the steps below:

First, download this repository using the version control system
[git](https://git-scm.com/). Type the following command into a terminal:

```bash
$ git clone https://github.com/richardkoehler/paper-otcs
```

Use the package manager
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) to set up a
new working environment. To do so, in your command line navigate to the location
where this repository is stored on your machine and type:

```bash
$ conda env create -f env.yml
```

This will create a new conda environment called `paper-otcs` and install Python including all necessary packages. Then activate the environment:

```bash
$ conda activate paper-otcs
```

Due to version conflicts between packages, when running the scripts for plotting counts, you must install and activate a second environment instead: 

```bash
$ conda env create -f env_plot_counts.yml
$ conda activate paper-otcs-plot-counts
```

## Questions
In case of questions related to the code feel free to contact me on GitHub.
