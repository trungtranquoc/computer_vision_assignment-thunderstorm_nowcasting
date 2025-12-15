import json
import os

COLOR_LEGEND_PATH = 'data/legend/color_dbz.json'

FILLER_COLOR_LIST = [
    [0, [255, 255, 255]],
    [1, [200, 200, 200]],
    [2, [150, 150, 150]],
]

os.makedirs(os.path.dirname(COLOR_LEGEND_PATH), exist_ok=True)
if not os.path.exists(COLOR_LEGEND_PATH):
    with open(COLOR_LEGEND_PATH, 'w') as f:
        json.dump(FILLER_COLOR_LIST, f, indent=2)

with open(COLOR_LEGEND_PATH) as f:
    list_color = json.load(f)

SORTED_COLOR = sorted(
    {tuple(color[1]): color[0] for color in list_color}.items(),
    key=lambda item: item[1]
)