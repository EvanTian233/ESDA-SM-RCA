"""
Exploratory Spatial Data Analysis for Smart Mobility Root Cause Analysis

Chapter 3.2: Data Augmentation with Coordinates
Chapter 3.3: Exploratory Spatial Data Analysis (ESDA)

Yifang Tian & Yaming Liu

"""


import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime

np.random.seed(42)

# Load data
data_pod_transmit = np.load(
    'C:/Users/yamin/Desktop/作业/1786/project/20240115_matrix/20240115/latency/pod_level_data_received_bandwidth.npy',
    allow_pickle=True).item()
data_pod_transmit = pd.DataFrame(data_pod_transmit)

# Process time values
time_values = pd.to_datetime([int(t) for t in data_pod_transmit.loc['time', 'book_info']], unit='s')
print('Length of time_values:', len(time_values))

pod_names = data_pod_transmit.loc['Pod_Name'][0]
print('Length of pod_names:', len(pod_names))

sequences_data = np.array(data_pod_transmit.loc['Sequence'][0])
print('sequences_data shape:', sequences_data.shape)

num_time_points, num_columns = sequences_data.shape
num_pod_names = len(pod_names)

if num_columns != num_pod_names:
    print('Mismatch detected between number of columns in sequences_data and length of pod_names.')
    print('First few values of the first column in sequences_data:')
    print(sequences_data[:5, 0])
    sequences_data = sequences_data[:, :-1]
    num_columns -= 1
    print('Removed the first column from sequences_data.')
    print('New sequences_data shape:', sequences_data.shape)

assert sequences_data.shape[1] == len(pod_names), \
    "After adjustment, the number of columns in sequences_data and length of pod_names still do not match."

sequences = pd.DataFrame(sequences_data, columns=pod_names, index=time_values)
print('Created sequences DataFrame with shape:', sequences.shape)

# Slice the data based on time for visualization
start_time = pd.to_datetime('2024-01-14 06:30:00')
end_time = pd.to_datetime('2024-01-14 08:30:00')
sequences = sequences[(sequences.index >= start_time) & (sequences.index <= end_time)]
print('Filtered sequences DataFrame with shape:', sequences.shape)

# Resample into 5 minutes
sequences_resampled = sequences.resample('5T').sum()
print('Resampled sequences DataFrame with shape:', sequences_resampled.shape)

pod_info_list = []

for pod_name in sequences_resampled.columns:
    pod_sequence = sequences_resampled[pod_name]
    prev_values = pod_sequence.shift(1).replace(0, 1)
    change_rate = (pod_sequence - prev_values) / prev_values
    change_rate.iloc[0] = 0
    change_rate.replace([np.inf, -np.inf], np.nan, inplace=True)
    change_rate.fillna(0, inplace=True)
    change_rate = change_rate.values

    if pod_name == 'scenario10-malware-deployment-57db4df9f4-p8h7h':
        zone = 1
    elif pod_name.startswith("scenario10-bot-deployment-69bbf69f44"):
        zone = 2
    else:
        zone = 3

    pod_info_list.append({
        'pod_name': pod_name,
        'zone': zone,
        'change_rate': change_rate,
        'sequence_sum': pod_sequence.values
    })

pod_info_df = pd.DataFrame(pod_info_list)

# Cordinate for root cause node
center_lat = 43.6532
center_lon = -79.3832


"""
Data augmentation for pods
"""

# Generate cordinates in a circle
def generate_random_points_in_circle(center_lon, center_lat, radius_km, num_points):
    coords = []
    for _ in range(num_points):
        r = radius_km * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2*np.pi)
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        delta_lon = dx / (111.320 * np.cos(np.deg2rad(center_lat)))
        delta_lat = dy / 110.574
        lon = center_lon + delta_lon
        lat = center_lat + delta_lat
        coords.append((lon, lat))
    return coords


# Generate cordinates in a annulus
def generate_random_points_in_annulus(center_lon, center_lat, inner_radius_km, outer_radius_km, num_points):
    coords = []
    for _ in range(num_points):
        r = np.sqrt(np.random.uniform(inner_radius_km**2, outer_radius_km**2))
        theta = np.random.uniform(0, 2 * np.pi)
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        delta_lon = dx / (111.320 * np.cos(np.deg2rad(center_lat)))
        delta_lat = dy / 110.574
        lon = center_lon + delta_lon
        lat = center_lat + delta_lat
        coords.append((lon, lat))
    return coords

zone1_pods = pod_info_df[pod_info_df['zone'] == 1] # Root Cause Pod
zone2_pods = pod_info_df[pod_info_df['zone'] == 2] # Secondary infecting pods
zone3_pods = pod_info_df[pod_info_df['zone'] == 3] # Tertiary infecting pods

zone1_coords = [(center_lon, center_lat)]
radius_zone2 = 3
zone2_coords = generate_random_points_in_circle(center_lon, center_lat, radius_zone2, len(zone2_pods))
inner_radius_zone3 = 3
outer_radius_zone3 = 6
zone3_coords = generate_random_points_in_annulus(center_lon, center_lat, inner_radius_zone3, outer_radius_zone3, len(zone3_pods))

pod_info_df.loc[pod_info_df['zone'] == 1, ['lon', 'lat']] = zone1_coords
pod_info_df.loc[pod_info_df['zone'] == 2, ['lon', 'lat']] = zone2_coords
pod_info_df.loc[pod_info_df['zone'] == 3, ['lon', 'lat']] = zone3_coords


"""
Exploratory Spatial Data Analysis (ESDA)
"""

heat_data_sequence = []
time_index = sequences_resampled.index.strftime('%Y-%m-%d %H:%M:%S').tolist()

global_threshold_high = 5e7
global_max_value = sequences_resampled.max().max()

# Generate heatmap
for frame in range(len(sequences_resampled)):
    data = []
    for idx, row in pod_info_df.iterrows():
        sequence_sum_values = row['sequence_sum']
        current_value = sequence_sum_values[frame]

        current_value_capped = np.clip(current_value, None, global_threshold_high)

        current_value_log = np.log(current_value_capped + 1)

        if global_max_value > 0:
            norm_value = current_value_log / np.log(global_threshold_high + 1)
        else:
            norm_value = 0.0
        norm_value = float(norm_value)

        data.append([row['lat']+0.045, row['lon'], norm_value])
    heat_data_sequence.append(data)

print("Heatmap data (first 5 frames):")
for i in range(min(5, len(heat_data_sequence))):
    print(f"Frame {i}: {heat_data_sequence[i]}")

m_sequence = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Gradient of the heatmap
gradient = {
    0.0: '#0000FF',
    0.1: '#0040FF',
    0.2: '#0080FF',
    0.3: '#00BFFF',
    0.4: '#00FFFF',
    0.5: '#40FFBF',
    0.6: '#80FF80',
    0.7: '#BFFF40',
    0.8: '#FFFF00',
    0.9: '#FFBF00',
    1.0: '#FF0000'
}


HeatMapWithTime(
    data=heat_data_sequence,
    index=time_index,
    auto_play=True,
    max_opacity=0.8,
    min_opacity=0.2,
    use_local_extrema=False,
    radius=15,
    gradient=gradient,
).add_to(m_sequence)

m_sequence.save('toronto_pod_sequence_heatmap.html')

print('Sequence Heatmap saved as toronto_pod_sequence_heatmap.html')

m_change_rate = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Normalize the scale using log transform
def normalize_values(values):
    norm_values = np.zeros_like(values, dtype=float)
    values = np.array(values, dtype=float)
    current_value_capped = np.clip(values, None, global_threshold_high)
    current_value_log = np.log(current_value_capped + 1)
    norm_values[0] = current_value_log / np.log(global_threshold_high + 1)
    if norm_values[0] < 0:
        norm_values[0] = 0
    return norm_values

heat_data_change_rate = []

for frame in range(len(sequences_resampled)):
    data = []
    for idx, row in pod_info_df.iterrows():
        change_rate_values = row['change_rate']
        current_value = change_rate_values[frame]

        if frame < 8:
            norm_value = float(1)
        else:
            norm_value = normalize_values([current_value])[0]

        data.append([row['lat'] + 0.045, row['lon'], float(norm_value)])
    heat_data_change_rate.append(data)

print("Change rate heatmap data (first 5 frames):")
for i in range(min(5, len(heat_data_change_rate))):
    print(f"Frame {i}: {heat_data_change_rate[i]}")

HeatMapWithTime(
    data=heat_data_change_rate,
    index=time_index,
    auto_play=True,
    max_opacity=1.0,
    min_opacity=0.2,
    use_local_extrema=False,
    radius=15,
    gradient=gradient,
).add_to(m_change_rate)


m_change_rate.save('toronto_pod_change_rate_heatmap.html')

print('Change Rate Heatmap saved as toronto_pod_change_rate_heatmap.html')
