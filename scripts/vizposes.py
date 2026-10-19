import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# 
# Configuration
# 
JSON_DIR = "/path/to/sealect_root/unified/val_annotations"

X_AXIS = "yaw"
Y_AXIS = "pitch"
COLOR_AXIS = "roll"

GRID_ROWS = 4
GRID_COLS = 2

CIRCLE_SIZE = 80
EDGE_COLOR = "black"  # constant outline for all points
EDGE_WIDTH = 1.2
COLORMAP = "coolwarm"  # roll color spectrum

# 
# Load pose data
# 
data_by_dataset = defaultdict(lambda: {"x": [], "y": [], "c": []})
all_rolls = []

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue

    dataset_name = os.path.splitext(fname)[0].split("_")[-1]

    with open(os.path.join(JSON_DIR, fname), "r") as f:
        data = json.load(f)

    for ann in data.get("annotations", []):
        pose = ann.get("pose")
        if pose is None:
            continue

        x_val = pose.get(X_AXIS)
        y_val = pose.get(Y_AXIS)
        c_val = pose.get(COLOR_AXIS)

        if x_val is None or y_val is None or c_val is None:
            continue

        data_by_dataset[dataset_name]["x"].append(x_val)
        data_by_dataset[dataset_name]["y"].append(y_val)
        data_by_dataset[dataset_name]["c"].append(c_val)
        all_rolls.append(c_val)

all_rolls = np.array(all_rolls)

# 
# Color normalization
# 
vmin = np.min(all_rolls)
vmax = np.max(all_rolls)
norm = Normalize(vmin=vmin, vmax=vmax)

# 
# Plotting
# 
datasets = sorted(data_by_dataset.keys())
fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 16), sharex=True, sharey=True)
axes = axes.flatten()

for ax, ds in zip(axes, datasets):
    x = np.array(data_by_dataset[ds]["x"])
    y = np.array(data_by_dataset[ds]["y"])
    c = np.array(data_by_dataset[ds]["c"])

    ax.scatter(
        x,
        y,
        s=CIRCLE_SIZE,
        c=c,
        cmap=COLORMAP,
        norm=norm,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_WIDTH
    )
    ax.set_title(ds)
    ax.grid(True, linestyle="--", alpha=0.3)

# Turn off unused subplots
for ax in axes[len(datasets):]:
    ax.axis("off")

# fig.suptitle("Pose Distribution per Dataset (Yaw–Pitch with Roll Color Encoding)", fontsize=16)
# fig.supxlabel("Yaw (degrees)")
# fig.supylabel("Pitch (degrees)")

# 
# Colorbar
# 
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
sm = ScalarMappable(cmap=COLORMAP, norm=norm)
sm.set_array(all_rolls)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Roll (degrees)", rotation=270, labelpad=18)

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()
