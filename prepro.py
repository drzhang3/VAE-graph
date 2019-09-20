import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("hangzhou_corp_48h_status_500.csv")
print(data.head())
time = list(data.time)
num = data.num
plt.plot(num)
plt.show()

