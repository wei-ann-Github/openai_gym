import os
import sys

import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
import spacy
import torch
import torchtext

ROOT = Path('../')
DATADIR = ROOT / 'data'
MODELDIR = ROOT / 'models'

sys.path.append(ROOT.__str__())
