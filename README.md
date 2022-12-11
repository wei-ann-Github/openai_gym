# openai_gym

For learning RL using the openai gymnasium (aka gym) framework

## Repo Structure

The branches follow the git-flow framework.

**Permanent Branches**
- master  # the most stable code.
- develop  # code that is almost stable, but not quite. Merge to `master` branch when ready.

**Temporary Branches**
- feature  # Feature that is still developing. To be merge into the `develop` branch when completed.
- hotfix
- bugfix
- release

## Directory Structure

ROOT
.
+-- environment.yml  # conda environment specifications. install with `conda env create -f enviroment.yml`
+-- requirements.txt  # python library requirements. install with `pip install -r requirements.txt`
+-- README.md
+-- .env.example  # your environment variables.
+-- Dockerfile
+-- .gitignore
+-- .dockerignore
+-- _data
|   +-- _raw  # raw data comes here.
|   +-- _interim  # data files there are preprocessed and persisted comes here.
|   +-- _processed  # the final preprocessed data that is ready for input into the model.
+-- _src  # all reusable python scripts and helper functions comes here.
|   +-- __init__.py
|   +-- utils.py  # utility helper functions come here.
+-- _models  # saved model comes here.
+-- _nb  # all your notebooks in this directory.
|   +-- config.py  # standard notebook imports
|   +-- template.ipynb  # template for all notebooks. Do give a version number to notes book serving the same function.
+-- _images  # Visualizations you created comes here.
+-- _references  # Papers that you referred to for the projects.
+-- _tests
|   +-- __init__.py
|   +-- conftest.py
|   +-- _unit
|      +-- __init__.py
|   +-- _functional
|      +-- __init__.py

## Instructions

### Env setup

```
conda env create -f enviroment.yaml
conda activate openai_gym
pip install -r requirements.txt
```


## Version

TBA
