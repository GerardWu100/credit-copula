"""Notebook section: credit risk portfolio analysis comparing dependence structures."""

# Every step file runs inside one shared namespace, so this first step imports
# for the whole pipeline. ``pd``, ``stats``, and ``norm`` look unused here but
# are read by later steps; do not let an automatic "unused import" cleanup
# remove them.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401  used by steps 02, 05, 06, 09
import seaborn as sns
from scipy import stats  # noqa: F401  used by steps 05, 06
from scipy.stats import norm  # noqa: F401  used by steps 04, 09

# Fixed seed so notebook reruns and smoke tests are reproducible.
np.random.seed(111111)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
