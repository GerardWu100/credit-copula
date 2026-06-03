"""Notebook section: credit risk portfolio analysis comparing dependence structures."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

# Fixed seed so notebook reruns and smoke tests are reproducible.
np.random.seed(111111)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
