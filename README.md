# Install
I included a requirements.txt file. I've had cross-platform issues sharing conda or pip requirements, the below should work
```
conda create -n march_madness python=3.7 pandas jupyter seaborn matplotlib scikit-learn graphviz# todo create pip freeze or conda versions
. activate march_madness
#conda install -c conda-forge featuretools  #version 0.6 on conda, current is 0.6.1
pip install featuretools
```
# Data
Download all here and extract to data/ dir https://www.kaggle.com/c/mens-machine-learning-competition-2019/data
