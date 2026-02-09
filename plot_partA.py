import numpy as np
import matplotlib.pyplot as plt

data = np.load("partA_results.npz")

times = data["times"]
start_length = data["start_lengths"]
peak_mems = data["peak_mems"]

x = start_length

plt.figure()
plt.plot(x, times, marker="o")
plt.xlabel("start length")
plt.ylabel("Total inference time (s)")
plt.title("Total inference time")
plt.show()

plt.figure()
plt.plot(x, peak_mems, marker="o")
plt.xlabel("start length")
plt.ylabel("Peak GPU memory (MB)")
plt.title("Peak GPU memory usage")
plt.show()
