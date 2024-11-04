import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_pod_transmit = np.load(
    'C:/Users/yamin/Desktop/作业/1786/project/20240115_matrix/20240115/latency/pod_level_data_transmit_bandwidth.npy',
    allow_pickle=True).item()
data_pod_transmit = pd.DataFrame(data_pod_transmit)

time_values = pd.to_datetime(
    [int(t) for t in data_pod_transmit.loc['time', 'book_info']], unit='s')
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
    print('Removed the last column from sequences_data.')
    print('New sequences_data shape:', sequences_data.shape)

assert sequences_data.shape[1] == len(pod_names), \
    "After adjustment, the number of columns in sequences_data and length of pod_names still do not match."

sequences = pd.DataFrame(sequences_data, columns=pod_names, index=time_values)
print('Created sequences DataFrame with shape:', sequences.shape)

plt.figure(figsize=(14, 8))
for pod_name in sequences.columns:

    if not pod_name.startswith("scenario10-bot-deployment-69bbf69f44"):
    # if pod_name == 'scenario10-malware-deployment-57db4df9f4-p8h7h':
        sequence_values = sequences[pod_name].resample('5min').sum()
        test = sequence_values.index
        test_1 = sequence_values.values
        plt.plot(sequence_values.index, sequence_values.values, label=pod_name)

plt.xlabel('Time')
plt.ylabel('Sum Sequence Value')
plt.title('Pod Level receive Bandwidth (5-Minute Average) over Time')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize='small')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
for pod_name in sequences.columns:
    # if not pod_name.startswith("scenario10-bot-deployment-69bbf69f44"):
    if pod_name == 'scenario10-malware-deployment-57db4df9f4-p8h7h':
        test = sequences[pod_name]
        sequence_values = sequences[pod_name].resample('5min').sum()
        test = sequence_values.values

        prev_values = sequence_values.shift(1).replace(0, 1)

        change_rate = (sequence_values - prev_values) / prev_values
        change_rate.replace([np.inf, -np.inf], np.nan, inplace=True)
        change_rate.fillna(0, inplace=True)
        test_1 = change_rate.values
        plt.plot(change_rate.index, change_rate.values, label=pod_name)

plt.xlabel('Time')
plt.ylabel('Change Rate')
plt.title('Pod Level Change Rate of Receive Bandwidth over Time')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize='small')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


