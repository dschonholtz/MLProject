"""
Plots the AUC of the data for channel and time voted data.
It does this due to a mistake when collecting data.

"""
import json
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import seaborn as sns


def channel_voted_results(raw_pre, raw_non, num_channels):
    """
    Collapses the results into channel voted results.
    :return: Channel_voted_pre_results, Channel_voted_non_results
    """
    channel_voted_pre = []
    channel_voted_non = []
    pre_idx = 0
    non_idx = 0
    for channel_count in num_channels:
        if pre_idx < len(raw_pre):
            avg = sum(raw_pre[pre_idx:pre_idx + channel_count]) / channel_count
            channel_voted_pre.append(avg)
            pre_idx += channel_count
        else:
            # handling non data
            avg = sum(raw_non[non_idx:non_idx + channel_count]) / channel_count
            channel_voted_non.append(avg)
            non_idx += channel_count

    return channel_voted_pre, channel_voted_non


def time_voted_results(channel_voted_pre, channel_voted_non, num_samples):
    """
    Collapses the results into time voted results.
    :return:
    """
    time_voted_pre = []
    time_voted_non = []
    for i in range(0, len(channel_voted_pre), num_samples):
        avg = sum(channel_voted_pre[i:i + num_samples]) / num_samples
        time_voted_pre.append(avg)
    for i in range(0, len(channel_voted_non), num_samples):
        avg = sum(channel_voted_non[i:i + num_samples]) / num_samples
        time_voted_non.append(avg)

    return time_voted_pre, time_voted_non


def parse_json(json_file_path):
    """
    Parses the json file and returns the results.
        results[pat_key] = {
        'epochs': [],
        'num_samples': 0,
        'non_results_files': [],
        'non_results': [], 0 or 1
        'pre_results_files': [],
        'pre_results': [] 0 or 1
    }
    :return:
    """
    with open(json_file_path) as json_file:
        result_obj = json.load(json_file)
    pat_key = json_file_path.split(os.sep)[-1].split('.')[0]
    results = result_obj[pat_key]

    # the non subset is the same size as the pre subset but there was an error when generating the results files
    # so we need to copy over the first section of the non subset to the end of the pre-subset so they are of equal size
    idx = 0
    new_pre = []
    new_non = []
    all_results = results['pre_results_files'] + results['non_results_files']
    expected_len = len(all_results) // 2
    for channel in results['channels_per_file']:
        if idx < expected_len:
            new_pre += all_results[idx:idx + channel]
        else:
            new_non += all_results[idx:idx + channel]
        idx += channel
    print('final_indx: ', idx)
    print('new_pre: ', len(new_pre))
    print('new_non: ', len(new_non))
    print('total Lengh: ', len(new_pre) + len(new_non))

    results['pre_results_files'] = new_pre
    results['non_results_files'] = new_non

    channel_voted_pre, channel_voted_non = channel_voted_results(
        results['pre_results_files'],
        results['non_results_files'],
        results['channels_per_file']
    )
    time_voted_pre, time_voted_non = time_voted_results(
        channel_voted_pre,
        channel_voted_non,
        9
    )

    return (new_pre, new_non), (channel_voted_pre, channel_voted_non), (time_voted_pre, time_voted_non)


def get_auc(channel_voted_non, channel_voted_pre, time_voted_non, time_voted_pre):
    """
    Calculates the AUC of the data for channel and time voted data.
    :return: A dictionary containing the AUC values for channel and time voted data
    """
    # Channel voted AUC calculation
    y_true = [0] * len(channel_voted_non) + [1] * len(channel_voted_pre)
    y_scores = channel_voted_non + channel_voted_pre
    fpr_channel, tpr_channel, _ = metrics.roc_curve(y_true, y_scores)
    roc_auc_channel = metrics.auc(fpr_channel, tpr_channel)

    # Time voted AUC calculation
    y_true = [0] * len(time_voted_non) + [1] * len(time_voted_pre)
    y_scores = time_voted_non + time_voted_pre
    fpr_time, tpr_time, _ = metrics.roc_curve(y_true, y_scores)
    roc_auc_time = metrics.auc(fpr_time, tpr_time)

    return {'channel_voted_auc': roc_auc_channel, 'time_voted_auc': roc_auc_time,
            'fpr_channel': fpr_channel, 'tpr_channel': tpr_channel,
            'fpr_time': fpr_time, 'tpr_time': tpr_time}


def plot_all_auc(aucs):
    """
    Plot all of the channel voted ROC curves on one chart.
    Plot all of the time voted ROC curves on another chart.
    Plot the average AUC values for channel and time voted data on a bar chart.
    :param aucs:
    :return:
    """
    # Create a directory to save the plots
    plot_directory = 'plots'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    # Channel Voted ROC curves
    plt.figure()
    for key, auc_data in aucs.items():
        plt.plot(auc_data['fpr_channel'], auc_data['tpr_channel'],
                 label=f'{key} (AUC = {auc_data["channel_voted_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Channel Voted Receiver Operating Characteristic CNN to LSTM')

    # Move the legend to the right of the chart and adjust the height
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='x-small')
    plt.gcf().subplots_adjust(right=0.75)

    # plt.show()
    plt.savefig(os.path.join(plot_directory, 'channel_voted_roc_curves_CNNtoLSTM.png'), bbox_inches='tight')
    plt.close()

    # Time Voted ROC curves
    plt.figure()
    for key, auc_data in aucs.items():
        plt.plot(auc_data['fpr_time'], auc_data['tpr_time'],
                 label=f'{key} (AUC = {auc_data["time_voted_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Time Voted Receiver Operating Characteristic CNN to LSTM')

    # Move the legend to the right of the chart and adjust the height
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='x-small')
    plt.gcf().subplots_adjust(right=0.75)
    plt.savefig(os.path.join(plot_directory, 'time_voted_roc_curves_CNNTOLSTM.png'), bbox_inches='tight')
    # plt.close()

    # plt.show()
    # Average AUC values bar chart
    labels = list(aucs.keys())
    channel_voted_aucs = [auc_data['channel_voted_auc'] for _, auc_data in aucs.items()]
    time_voted_aucs = [auc_data['time_voted_auc'] for _, auc_data in aucs.items()]
    # Calculate average AUC values for all patients
    avg_channel_voted_auc = sum(channel_voted_aucs) / len(channel_voted_aucs)
    avg_time_voted_auc = sum(time_voted_aucs) / len(time_voted_aucs)

    # Add average AUC values for all patients to the bar chart
    labels.append("Average")
    channel_voted_aucs.append(avg_channel_voted_auc)
    time_voted_aucs.append(avg_time_voted_auc)
    print(f'Average AUC for channel voted data: {avg_channel_voted_auc:.2f}')
    print(f'Average AUC for time voted data: {avg_time_voted_auc:.2f}')

    width = 0.35  # the width of the bars
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, channel_voted_aucs, width, label='Channel Voted')
    rects2 = ax.bar(x + width/2, time_voted_aucs, width, label='Time Voted')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC Values')
    ax.set_title('Average AUC Values for Channel and Time Voted Data for CNN into LSTM')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.1, 0.05))
    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='lower left')

    fig.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plot_directory, 'average_auc_values_LSTM_bar_chart.png'), bbox_inches='tight')
    plt.close()



def main():
    """
    Plots the statistics of the predicted data.

    Accuracy on all data, pre-seiz data, and non-seiz data.

    Accuracy with channel voting
    Accuracy with Time voting

    Sensitivity and Specificity

    :return:
    """
    print('running')
    results_dir = Path("results") / "resultsLSTM"
    all_json_files = results_dir.glob("*.json")
    all_stats = dict()
    aucs = dict()
    for json_file in all_json_files:
        print(f"Processing: {json_file}")
        (new_pre, new_non), (channel_voted_pre, channel_voted_non), (time_voted_pre, time_voted_non) = parse_json(
            str(json_file))
        aucs[json_file.stem] = get_auc(channel_voted_non, channel_voted_pre, time_voted_non, time_voted_pre)

    plot_all_auc(aucs)


if __name__ == '__main__':
    main()