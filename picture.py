import numpy as np
import matplotlib.pyplot as plt

data = np.load('my_file.npy', allow_pickle=True)
d = data.item()
x = list(d.keys())
x = list(map(str, x))
y = list(d.values())
plt.bar(x, y)
plt.title('threshold =16')
plt.xlabel('measurement outcome')
plt.ylabel('probability')
plt.xticks(rotation=315)
plt.show()
