import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

T = 1000

# Create figure and axes
fig, axes = plt.subplots(nrows=2, figsize=(16, 9))

df = pd.DataFrame({
                  'time': np.arange(T),
                  'success': np.random.choice([True, False], size=T),
                  'choice': np.random.choice([0, 1], size=T)})
sns.swarmplot(x='time', y='choice',
              data=df, ax=axes[0], orient="h")

sns.swarmplot(x='time', y='success',
              data=df, ax=axes[1], palette={True: "green",
                                            False: "red"}, orient="h")

plt.show()