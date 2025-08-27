# AI4ALL Drone Control Project

Source code and coding exercises for drone navigation project: designing a simple grid-world maze environment, controlling CrazyFlie drone platforms, implementing Q-Learning and DQN agents for progressively more complex tasks, and integrating drone control and RL path planning.

This repository contains and extends helper code originally provided in Princeton's MAE345 Intro to Robotics Class (the code for initializing drones, dependencies, etc). The original contributions are the notebooks in the top level directory. The following installation instructions are pulled from the instructions provided in the MAE345 repository.

- All Notebooks are paired: both the base notebook and the completed 'instructor version' have been provided.

## Install Instructions 

Included in this repository is a conda environment named `env-mae345.yml`. For the unfamiliar, [conda](https://docs.conda.io/en/latest/) (short for Anaconda) is a tool for managing Python environments --- collections of software and libraries for developing Python programs. Conda environments make it very easy to reproduce and share code with other developers (in this case between the students and AIs).

To install the environment, do the following:

1. Download and install [Anaconda](https://www.anaconda.com/products/individual).

2. On Mac and Linux, open the terminal. Navigate to where this repository has been downloaded (entering `ls` will list the files and directories accessible from your current directory and `cd <name>` will change you to the `<name>` directory) and run `conda env create -f env-mae345.yml`. Accept any of the prompted changes. On Windows, do the same, use the Anaconda Prompt application that should be present in your start menu (on Windows you need to use `dir` to list the contents of a directory instead of `ls`).

## Working on Assignments

To work on an assignment, open the terminal (on Windows you need to use the same Anaconda Prompt application you used to install the environment) and navigate to the directory containing this repository. Enter the command `conda activate mae345` to load the environment. Then run either `jupyter lab` or `jupyter notebook`. Both launch an interface for editing and running Python scripts in your browser. Some notebooks (those not dealing with drone control) are also fully functional in Google Colab. Follow the instructions within the notebook to complete the assignment.

