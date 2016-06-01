[![Build Status](https://travis-ci.org/mgraffg/EvoDAG.svg?branch=master)](https://travis-ci.org/mgraffg/EvoDAG)

[![PyPI version](https://badge.fury.io/py/EvoDAG.svg)](https://badge.fury.io/py/EvoDAG)

[![Coverage Status](https://coveralls.io/repos/github/mgraffg/EvoDAG/badge.svg?branch=master)](https://coveralls.io/github/mgraffg/EvoDAG?branch=master)

# EvoDAG

Evolving Directed Acyclic Graph (EvoDAG) is a steady-state Genetic Programming system
with tournament selection. The main characteristic of EvoDAG is that
the genetic operation is performed at the root. EvoDAG was inspired
by the geometric semantic crossover proposed by 
[Alberto Moraglio](https://scholar.google.com.mx/citations?user=0y4XRI0AAAAJ&hl=en&oi=ao)
_et al._ and the implementation performed by
[Leonardo Vanneschi](https://scholar.google.com.mx/citations?user=uR5K07QAAAAJ&hl=en&oi=ao)
_et al_.

# Example using command line

Let us assume one wants to create a classifier of iris dataset. The
first step is to download the dataset from the UCI Machine Learning
Repository

```bash   
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

In order to train the EvoDAG using a population of 10 individuals,
using early stopping to 11, sampling 100 different parameter configurations, creating
an ensemble of 12, and using 4 cores then the following command is used:

```bash   
~/.local/bin/EvoDAG -e 10 -p 11 -r 100 -u 4 -n 12 iris.data
```

The EvoDAG ensemble is stored in iris.evodag.gz. 

Now that the ensemble has been initialized one can predict a test set
and store the output in file called output.csv using the following command.

```bash   
~/.local/bin/EvoDAG -m iris.evodag.gz -t iris.data -o output.csv
```


## Install EvoDAG

* Install using pip  
```pip install EvoDAG```

### Using the source code
* Clone the repository  
```git clone  https://github.com/mgraffg/EvoDAG.git```
* Install the package as usual  
```python setup.py install```
* To install only for the use then  
```python setup.py install --user```


