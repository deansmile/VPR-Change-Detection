import matplotlib.pyplot as plt
import numpy as np

# Method names
methods = ['RSCD', 'C-3PO', 'GeSCF']

# Original and improved metrics
f1_original = [0.411, 0.0944, 0.162]
f1_with_lang = [0.522, 0.5741, 0.571]
iou_original = [0.310, 0.0726, 0.102]
iou_with_lang = [0.411, 0.4738, 0.486]

# Compute performance gains
f1_increase = [with_lang - orig for orig, with_lang in zip(f1_original, f1_with_lang)]
iou_increase = [with_lang - orig for orig, with_lang in zip(iou_original, iou_with_lang)]

x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot F1
f1_bars = ax.bar(x - width/2, f1_original, width, label='F1 Original', color='skyblue')
f1_gain_bars = ax.bar(x - width/2, f1_increase, width, bottom=f1_original, label='F1 Gain', color='blue')

# Plot IoU
iou_bars = ax.bar(x + width/2, iou_original, width, label='IoU Original', color='lightgreen')
iou_gain_bars = ax.bar(x + width/2, iou_increase, width, bottom=iou_original, label='IoU Gain', color='green')

# Labels and formatting
ax.set_ylabel('Performance')
ax.set_title('Model Performance with and without Language Module on our dataset')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Model Performance Data", dataframe=
#     {
#         "Method": methods,
#         "F1 Original": f1_original,
#         "F1 With Language": f1_with_lang,
#         "IoU Original": iou_original,
#         "IoU With Language": iou_with_lang
#     })

plt.show()
