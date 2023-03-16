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


def get_accuracy(pre_siez_results, non_siez_results):
    """
    Gets the accuracy from the given results.
    :return:
    """
    total_results = len(pre_siez_results) + len(non_siez_results)
    correct_results = pre_siez_results.count(1) + non_siez_results.count(0)
    return correct_results / total_results


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

    raw_stats = get_stats(results['pre_results_files'], results['non_results_files'])
    print(f"Raw Stats: {raw_stats}")
    channel_voted_pre, channel_voted_non = channel_voted_results(
        results['pre_results_files'],
        results['non_results_files'],
        results['channels_per_file']
    )
    channel_voted_stats = get_stats(channel_voted_pre, channel_voted_non)
    print(f"Channel Voted Stats: {channel_voted_stats}")
    time_voted_pre, time_voted_non = time_voted_results(
        channel_voted_pre,
        channel_voted_non,
        5
    )
    time_voted_stats = get_stats(time_voted_pre, time_voted_non)
    print(f"Time Voted Stats: {time_voted_stats}")

    return raw_stats, channel_voted_stats, time_voted_stats


def plot_metrics(patients, raw_stats, title):
    # heights of bar charts
    raw_stats = np.array(raw_stats)
    accuracy = raw_stats[:, 0]
    sensitivity = raw_stats[:, 1]
    specificity = raw_stats[:, 2]
    # Set position of bar on X axis
    barWidth = 0.25
    br1 = np.arange(len(accuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, accuracy, color='r', width=barWidth,
            edgecolor='grey', label='accuracy')
    plt.bar(br2, sensitivity, color='g', width=barWidth,
            edgecolor='grey', label='sensitivity')
    plt.bar(br3, specificity, color='b', width=barWidth,
            edgecolor='grey', label='specificity')
    plt.xticks([r + barWidth for r in range(len(accuracy))],
               patients, rotation=50, ha='right', fontsize=8)

    # don't let the xticks get cut off at the bottom by making the graph larger
    plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='lower left')
    plt.title(title)
    fig1 = plt.gcf()
    plt.draw()
    print('AVG Accuracy: ', np.average(accuracy))
    print('AVG Sensitivity: ', np.average(sensitivity))
    print('AVG Specificity: ', np.average(specificity))
    # fig1.savefig(f"{title}.png", dpi=100)
    plt.show()


def plot_false_positive_rates(patients, raw_stats, title, samples_per_hour):
    # heights of bar charts
    raw_stats = np.array(raw_stats)
    false_positive_rate = raw_stats[:, 3]
    fp_per_hour = false_positive_rate * samples_per_hour
    # Set position of bar on X axis
    barWidth = 0.25
    br1 = np.arange(len(fp_per_hour))

    # Make the plot
    plt.bar(br1, fp_per_hour, color='r', width=barWidth,
            edgecolor='grey', label='false positives / hour')
    plt.xticks([r + barWidth for r in range(len(false_positive_rate))],
               patients, rotation=50, ha='right', fontsize=8)

    # print the average false positive / hour value for all patients
    print(f"Average false positives / hour: {np.mean(fp_per_hour)}")
    print(f'Median False positives / hour: {np.median(fp_per_hour)}')

    # don't let the xticks get cut off at the bottom by making the graph larger
    plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='upper left')
    plt.title(title)
    fig1 = plt.gcf()
    plt.draw()
    # fig1.savefig(f"{title}.png", dpi=100)
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
    all_stats = dict()
    for json_file in all_json_files:
        print(f"Processing: {json_file}")
        all_stats[str(json_file).split(os.sep)[-1].split('.')[0]] = parse_json(str(json_file))

    # save all_stats to a json file
    with open('LSTM_ALL_STATS.json', 'w') as outfile:
        json.dump(all_stats, outfile)

    all_stats = json.load(open('LSTM_ALL_STATS.json'))
    # plot the results
    raw_stats = [x[0] for _, x in all_stats.items()]
    channel_voted_stats = [x[1] for _, x in all_stats.items()]
    time_voted_stats = [x[2] for _, x in all_stats.items()]
    patients = [x for x in all_stats.keys()]
    model_name = 'CNN Transformer'
    plot_metrics(patients, raw_stats, f'Raw Results (No Voting), {model_name}')
    plot_metrics(patients, channel_voted_stats, f'Channel Voted Results, {model_name}')
    plot_metrics(patients, time_voted_stats, f'Time Voted Results 5 Sample Window, {model_name}')


    # plot false positive rates
    # plot_false_positive_rates(patients,
    #                           raw_stats,
    #                           f'False Positive Rates per Hour Raw Results (No Voting), {model_name}',
    #                           3600 / 4)
    plot_false_positive_rates(patients,
                                channel_voted_stats,
                                f'False Positives per Hour Channel Voted, {model_name}',
                                3600 / 4)
    plot_false_positive_rates(patients,
                                time_voted_stats,
                                f'False Positives per Hour Channel&Time Voted 5 Sample window, {model_name}',
                                3600 / (4 * 5))


if __name__ == '__main__':
    main()