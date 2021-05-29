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
sys.path.append(ROOT.__str__())

from src import *

DATADIR = ROOT / 'data'
MODELDIR = ROOT / 'models'
