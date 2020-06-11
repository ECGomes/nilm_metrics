from metrics.energy_estimation import MetricsEE
import pandas as pd
import numpy as np


def aux_group_interval(data, options):
    if options == 'None':
        temp_dates = data.index

        temp_list = []
        for i in temp_dates:
            temp_list.append(i)

        return temp_list
    elif options == 'Min':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month,
                                   data.index.day,
                                   data.index.hour,
                                   data.index.minute])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}-{:02d} {:02d}:{:02d}'.format(i[0], i[1], i[2], i[3], i[4]))

        return temp_list
    elif options == 'Hour':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month,
                                   data.index.day,
                                   data.index.hour])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}-{:02d} {:02d}'.format(i[0], i[1], i[2], i[3]))

        return temp_list
    elif options == 'Day':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month,
                                   data.index.day])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}-{:02d}'.format(i[0], i[1], i[2]))

        return temp_list
    elif options == 'Week':
        temp_dates = data.groupby(pd.Grouper(freq='W-Mon'))

        temp_list = []

        last_date = list(temp_dates.groups.keys())[len(list(temp_dates.groups.keys())) - 2]
        last_date = last_date.strftime('%Y-%m-%d')

        for i in list(temp_dates.groups.keys())[:-1]:
            temp_start = i.strftime('%Y-%m-%d')

            if temp_start != last_date:
                temp_end = i + pd.DateOffset(weeks=1) - pd.DateOffset(days=1)
                temp_end = temp_end.strftime('%Y-%m-%d')

                temp_list.append((temp_start, temp_end))

            else:
                temp_list.append((temp_start,))

        return temp_list
    elif options == 'Month':
        temp_dates = data.groupby([data.index.year,
                                   data.index.month])

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}-{:02d}'.format(i[0], i[1]))

        return temp_list
    elif options == 'Year':
        temp_dates = data.groupby(data.index.year)

        temp_list = []
        for i in temp_dates.groups.keys():
            temp_list.append('{}'.format(i))

        return temp_list
    else:
        print('Option not valid!')
        return


def check_metrics(df_gt, df_pred, metric_list, start_date='None', end_date='None', interval='None'):
    """
    :param df_gt: Ground-truth pandas DataFrame
    :param df_pred: Predictions on a pandas DataFrame
    :param start_date: Starting date for metric calculations
    :param end_date: Ending date for metric calculations
    :param metric_list: Set of metrics to calculate
    :param interval: Interval to use while calculating metrics: 'None', 'Min', 'Hour', 'Day', 'Week', 'Year'
    :return: Dictionary containing metrics for the interval specified
    """

    # Check available columns based on the GT frame
    col_list = []
    for col in df_pred.columns:
        if col in df_gt.columns:
            col_list.append(col)

    # Get the data within the time frame
    temp_gt = df_gt[col_list]
    temp_pred = df_pred[col_list]
    temp_start = 'None'
    temp_end = 'None'

    if start_date != 'None':
        temp_start = pd.to_datetime(start_date)
    if end_date != 'None':
        temp_end = pd.to_datetime(end_date)

    if start_date == 'None' and end_date == 'None':
        pass

    elif start_date == 'None' and end_date != 'None':
        temp_gt = temp_gt[temp_gt.index < temp_end]
        temp_pred = temp_pred[temp_pred.index < temp_end]

    elif start_date != 'None' and end_date == 'None':
        temp_gt = temp_gt[temp_gt.index >= temp_start]
        temp_pred = temp_pred[temp_pred.index >= temp_start]

    elif start_date != 'None' and end_date != 'None':
        temp_gt = temp_gt[(temp_gt.index >= temp_start) & (temp_gt.index < temp_end)]
        temp_pred = temp_pred[(temp_pred.index >= temp_start) & (temp_pred.index < temp_end)]

    sampled_time = aux_group_interval(temp_gt, interval)
    sampled_gt = {}
    sampled_pred = {}
    for timeframe in sampled_time:
        sampled_gt[timeframe] = temp_gt[timeframe]
        sampled_pred[timeframe] = temp_pred[timeframe]

    # Go through a list of metrics to calculate
    column_results = {}
    metrics_class = MetricsEE()

    unique_metrics_v1 = np.unique(np.array(metric_list))
    if 'cep' in unique_metrics_v1:
        unique_metrics_v1 = unique_metrics_v1[unique_metrics_v1 != 'cep']

        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_c')
        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_co')
        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_cu')
        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_o')
        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_ozero')
        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_u')
        unique_metrics_v1 = np.append(unique_metrics_v1, 'cep_total')

    unique_metrics_v2 = unique_metrics_v1.copy()
    for metric in unique_metrics_v1:
        if metrics_class.checkFunction(metric):
            pass
        else:
            unique_metrics_v2 = unique_metrics_v2[unique_metrics_v2 != metric]

    for col in col_list:

        metrics_results = {}
        for calc in unique_metrics_v2:

            results_timeframe = {}
            for timeframe in sampled_time:
                results_timeframe[timeframe] = metrics_class.callFunction(calc,
                                                                          temp_gt[col][timeframe],
                                                                          temp_pred[col][timeframe])

            metrics_results[calc] = results_timeframe

        metrics_results = pd.DataFrame(metrics_results)
        metrics_results.index = pd.to_datetime(metrics_results.index)

        column_results[col] = metrics_results

    return column_results
