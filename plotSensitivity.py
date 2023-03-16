"""
Plots the statistics of the predicted data.

Accuracy on all data, pre-seiz data, and non-seiz data.

Accuracy with channel voting
Accuracy with Time voting

Sensitivity and Specificity

"""
import json
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


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
            if avg > 0.5:
                channel_voted_pre.append(1)
            else:
                channel_voted_pre.append(0)
            pre_idx += channel_count
        else:
            # handling non data
            avg = sum(raw_non[non_idx:non_idx + channel_count]) / channel_count
            if avg > 0.5:
                channel_voted_non.append(1)
            else:
                channel_voted_non.append(0)
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
        if avg > 0.5:
            time_voted_pre.append(1)
        else:
            time_voted_pre.append(0)
    for i in range(0, len(channel_voted_non), num_samples):
        avg = sum(channel_voted_non[i:i + num_samples]) / num_samples
        if avg > 0.5:
            time_voted_non.append(1)
        else:
            time_voted_non.append(0)

    return time_voted_pre, time_voted_non


def get_stats(pre_results, non_results):
    """
    Gets the sensitivity and specificity from the given results.
    :return:
    """
    accuracy = get_accuracy(pre_results, non_results)
    false_positives = non_results.count(1)
    false_negatives = pre_results.count(0)
    true_positives = pre_results.count(1)
    true_negatives = non_results.count(0)
    print(f'False positives: {false_positives}')
    false_positive_rate = false_positives / (false_positives + true_negatives)

    sensitivity = true_positives / (true_positives + false_negatives)

    specificity = true_negatives / (true_negatives + false_positives)

    return accuracy, sensitivity, specificity, false_positive_rate


def parse_json(json_file_path, time_vote_samples):
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
    pre_results = results['pre_results_files'] + results['non_results_files']
    expected_len = len(pre_results) // 2
    for channel in results['channels_per_file']:
        if idx < expected_len:
            new_pre += pre_results[idx:idx + channel]
        else:
            new_non += pre_results[idx:idx + channel]
        idx += channel
    print('final_indx: ', idx)
    print('new_pre: ', len(new_pre))
    print('new_non: ', len(new_non))
    print('total Length: ', len(new_pre) + len(new_non))

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
        time_vote_samples
    )

    return channel_voted_pre, time_voted_pre


def plot_sensitivity(time, sens_all_mean, sens_all_std, fit, time_samples, is_time=False):
    # Make the plot
    fontsizelabel = 12

    fig8, ax8 = plt.subplots()
    ax8.yaxis.grid()
    plt.errorbar(time, sens_all_mean, yerr=sens_all_std, color='g', ecolor='g', label='STD')
    plt.plot(time, sens_all_mean, color='#575DFF', label='Mean', zorder=5)
    plt.plot(time, fit, color='#F0CFA5', label='Polynomial of Best Fit', zorder=10)
    plt.axvline(0, color='#FF5769')
    plt.annotate("Seizure Onset", xy=(0, 30), xytext=(-20, 20), fontsize=fontsizelabel,
                 arrowprops=dict(facecolor='black'), )
    plt.yticks(np.arange(0, 110, step=5))
    plt.xticks(np.arange(-60, 10, step=10))
    plt.legend(loc='lower right')
    plt.ylabel('Sensitivity (%)')
    plt.ylim([0,100])
    plt.xlabel('Time (mins)')
    title = "CNN to LSTM Seizure Horizon Sensitivity\n With Channel Voting Average"
    if is_time:
        title = f"CNN to LSTM Seizure Horizon Sensitivity\n With Time Voting {time_samples} Samples Average"
    plt.title(title, fontweight='bold')
    file_name = 'CNNtoLSTM_eeg_channel_voted.png' if not is_time else f'CNNtoLSTMSens_eeg_time_voted{time_samples}Time.png'
    plt.savefig(file_name)
    plt.show()

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
    time_samples = 5
    avg_non_seiz_channel = np.zeros((3600 // 4))
    avg_non_seiz_time = np.zeros((3600 // (4 * time_samples)))
    num_json_files = 0
    for json_file in all_json_files:
        num_json_files += 1
        print(f"Processing: {json_file}")
        parsed_channel, parsed_time = parse_json(str(json_file), time_samples)
        avg_non_seiz_channel += np.array(parsed_channel)[0:3600 // 4]
        avg_non_seiz_time += np.array(parsed_time)[0:3600 // (4 * time_samples)]

    avg_non_seiz_channel = avg_non_seiz_channel / num_json_files * 100
    avg_non_seiz_time = avg_non_seiz_time / num_json_files * 100
    sens_all_std_channel = np.nanstd(avg_non_seiz_channel, axis=0)
    sens_all_std_time = np.nanstd(avg_non_seiz_time, axis=0)

    time_channel = np.linspace(-60, 0, 900)
    time_time = np.linspace(-60, 0, 900 // time_samples)
    coef_channel = np.polyfit(time_channel, avg_non_seiz_channel, 2)
    fit_channel = np.polyval(coef_channel, time_channel)
    coef_time = np.polyfit(time_time, avg_non_seiz_time, 2)
    fit_time = np.polyval(coef_time, time_time)
    plot_sensitivity(time_channel, avg_non_seiz_channel, sens_all_std_channel, fit_channel, time_samples)
    plot_sensitivity(time_time, avg_non_seiz_time, sens_all_std_time, fit_time, time_samples, is_time=True)


if __name__ == '__main__':
    main()