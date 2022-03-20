import pandas as pd
import numpy as np
import statsmodels.stats.api as sms


def get_df(path):
    return pd.read_csv(path)


def get_formatted_df(df):
    df = df.drop(["releases_avg_downloads_per_day", "commits_avg_per_day",
                  "id", "repo_fullname", "created_at", "updated_at"], axis=1)
    df['commits_count'].fillna(1, inplace=True)
    df['commits_days_since_first'].fillna(df['commits_days_since_first'].mean(), inplace=True)
    df['commits_days_since_last'].fillna(df['commits_days_since_last'].mean(), inplace=True)
    # fill zero ['commits_total_lines_added', 'commits_total_lines_removed', 'commits_avg_added',
    #                   'commits_avg_removed', 'commits_avg_files_changed', 'commits_avg_message_length'
    #                   'commits_avg_per_day_real', 'commits_max_per_day']
    df.fillna(0, inplace=True)
    return df


def bootstrap(df, B=5000):
    metrics_l = []
    for metric in df:
        metrics_l.append(df[metric].to_list())
    metrics_l = np.array(metrics_l)
    metrics_l_means = []
    intervals = []
    for metric in metrics_l:
        local_means = []
        for _ in range(B):
            x = np.random.choice(metric, size=len(metric), replace=True)
            local_means.append(x.mean())
        intervals.append(sms.DescrStatsW(local_means).tconfint_mean())
        metrics_l_means.append(local_means)
    temp = {key: values for key, values in zip(df.keys(), intervals)}
    df_means = pd.DataFrame(temp, columns=df.keys())
    return df_means
