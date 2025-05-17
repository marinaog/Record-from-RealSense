import csv
from collections import defaultdict

def parse_mocap_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = list(csv.reader(csvfile))

    # Header rows
    data_rows = reader[7:]  # Data starts at line 7 (index 6 is the header row)
    rigid_bodies = ["solid1", "solid2"]
    results = {rb: {} for rb in rigid_bodies} 

    for row in data_rows:
        frame = row[0]
        values = row[1:] 

        if any(v == '' or v is None for v in values):
            continue  # skip incomplete rows

        entry1 = {
            'time': values[0],
            'rotx': values[1],
            'roty': values[2],
            'rotz': values[3],
            'x': values[4],
            'y': values[5],
            'z': values[6],
            'error': values[7]
            }
        
        entry2 = {
            'time': values[0],
            'rotx': values[8],
            'roty': values[9],
            'rotz': values[10],
            'x': values[11],
            'y': values[12],
            'z': values[13],
            'error': values[14]
            }
        
        results["solid1"][frame] = entry1
        results["solid2"][frame] = entry2

    return results


import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator

def plot_y_vs_frame(data, solid_name="solid1"):
    frames = sorted(data[solid_name].keys(), key=lambda x: int(x))
    print(float(data[solid_name][frames[0]]['y']) )
    y_values = [float(data[solid_name][frame]['y']) for frame in frames]
    frame_numbers = [int(f) for f in frames]

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, y_values, label=f'{solid_name} - y', color='blue')

    plt.xlabel('Frame')
    plt.ylabel('Y Position')
    plt.title(f'Y Position vs. Frame Number for {solid_name}')
    plt.grid(True)
    plt.legend()

    # Use dynamic ticks that respond to zoom
    ax = plt.gca()
    ax.xaxis.set_major_locator(AutoLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# Usage
filename = 'recording.csv'
data = parse_mocap_csv(filename)

import pprint
for body, frames in data.items():
    print(f"\n{body}:")
    for frame in sorted(frames)[:3]:
        pprint.pprint({frame: frames[frame]})

# Plotting the data:
plot_y_vs_frame(data)
