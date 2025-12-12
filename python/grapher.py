import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

results_file = "../jobscripts/results.txt"

pagerank_mod_x = []
pagerank_mod_y = []
pagerank_base_x = []
pagerank_base_y = []

with open(results_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        name, x_str, y_str = line.split(",")
        x = int(x_str)
        y = float(y_str)

        if name == "pagerank_mod":
            pagerank_mod_x.append(x)
            pagerank_mod_y.append(y)
        elif name == "pagerank_base":
            pagerank_base_x.append(x)
            pagerank_base_y.append(y)

mod_sorted = sorted(zip(pagerank_mod_x, pagerank_mod_y))
base_sorted = sorted(zip(pagerank_base_x, pagerank_base_y))

pagerank_mod_x, pagerank_mod_y = zip(*mod_sorted)
pagerank_base_x, pagerank_base_y = zip(*base_sorted)

# Plot
plt.figure(figsize=(10, 6))
# plt.plot(pagerank_mod_x, pagerank_mod_y, marker="o", label="pagerank_mod")
plt.plot(pagerank_base_x, pagerank_base_y, marker="o", label="pagerank_base")

plt.yscale("log",base=2)
plt.xscale("log", base=2)

ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='both')


plt.xlabel("Number of Processes")
plt.ylabel("Solution Time")
plt.title("Pagerank Base Preformance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
