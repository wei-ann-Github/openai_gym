import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
import spacy
import torch
import torchtext


ROOT = Path('../')
DATADIR = ROOT / 'data'
MODELDIR = ROOT / 'models'

sys.path.append(ROOT.__str__())
