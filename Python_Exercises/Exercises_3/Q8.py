import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = input("Enter the CSV filename: ")
data = pd.read_csv(filename)

sns.pairplot(data)
plt.show()
