import numpy as np
import matplotlib.pyplot as plt

data = np.load("partA_results.npz")

times = data["times"]
per_token_times = data["per_token_times"]
peak_mems = data["peak_mems"]

x = range(1, len(times) + 1)

plt.figure()
plt.plot(x, times, marker="o")
plt.xlabel("Sample index")
plt.ylabel("Total inference time (s)")
plt.title("Total inference time")
plt.show()

plt.figure()
plt.plot(x, per_token_times, marker="o")
plt.xlabel("Sample index")
plt.ylabel("Per-token time (s)")
plt.title("Per-token inference time")
plt.show()

plt.figure()
plt.plot(x, peak_mems, marker="o")
plt.xlabel("Sample index")
plt.ylabel("Peak GPU memory (MB)")
plt.title("Peak GPU memory usage")
plt.show()
