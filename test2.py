%%writefile preprocess_data.py
import os
import sys
import logging
import boto3
from dataclasses import dataclass, field
from typing import List
from datetime import datetime, timedelta
import datetime as dt

import pandas as pd
import numpy as np

from io import StringIO

import requests
from kfp.v2.dsl import component, Input, Output, OutputPath, Dataset, Model, Metrics, Artifact
from kfp.v2 import dsl
@component(
    packages_to_install=['pandas', 'numpy==1.19.2', 'scikit-learn', 'boto3'],
)
def preprocess_data(tar_data: Input[Dataset], tags_data: Input[Dataset], config_art: Input[Artifact],
                    X_train_art: Output[Dataset], X_test_art: Output[Dataset], 
                    y_train_art: Output[Dataset], y_test_art: Output[Dataset], 
                    scaler_art: Output[Artifact], output_tags_pid_art: Output[Artifact],
                    output_tags_idx_art: Output[Artifact], mapping_pid_to_name_art: Output[Artifact],
                    metrics: Output[Metrics]):
    import warnings
    from io import BytesIO
    from sklearn.model_selection import train_test_split
    from sklearn.exceptions import DataConversionWarning
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    import os
    import logging
    import boto3

    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from io import StringIO
    AWS_ACCESS_KEY_ID="AKIAZTN6P4WBRRX2YXAG"
    AWS_SECRET_ACCESS_KEY="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"
    class PreprocessData():

        def __init__(self, df, outlier_kind, analog_tags_pid, digital_tags_pid, physical_thresholds, n_digitals) -> None:
            self.dataset = df
            self.outlier_kind = outlier_kind
            self.analog_tags_pid = analog_tags_pid
            self.digital_tags_pid = digital_tags_pid
            self.physical_thresholds = physical_thresholds
            self.n_digitals = n_digitals

        def get_dataset(self):
            """return dataset
            @return
            df: pd.DataFrame
                preprocessed data
            """
            return self.dataset

        def drop_string(self):
            """cleaning dataset from string values
            @return
            df: pd.DataFrame
                preprocessed data
            """
            try:
                for i, col in enumerate(self.dataset.columns.values):
                    self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')#.notnull()
                self.dataset.dropna(inplace=True)
                self.dataset = self.dataset.astype('float32')
            except Exception as e:
                print(e)
            return self.dataset

        def drop_error(self, filter_thred=10000000000):
            """filter dataset on treshold
            @return
            df: pd.DataFrame
                preprocessed data
            """
            for i, pid in enumerate(self.dataset.columns):
                if pid in self.digital_tags_pid:
                    self.dataset.loc[self.dataset[pid] > 1, pid] = np.NaN
                    self.dataset.loc[self.dataset[pid] < 0, pid] = np.NaN
                else:
                    self.dataset.loc[self.dataset[pid] > filter_thred, pid] = np.NaN
            self.dataset.dropna(inplace=True)
            return self.dataset

        def Mahala_distantce(self, x, mean, cov):
            """
            calculate Mahalanobis distance
            @parameters
            x    : np.array
            mean : np.array
            cov  : np.array
            @return
            d    : np.array
                Mahalanobis distance
            """
            d = np.dot(x - mean, np.linalg.pinv(cov))
            d = np.dot(d, (x - mean).T)
            return d

        def cal_model_MD(y_error, mean_model, cov_model):
            """calculating MD error
            @return
            model_MD: list
            """
            model_MD = []
            for i in range(y_error.shape[0]):
                model_MD.append(self.Mahala_distantce(y_error[i, :], mean_model, cov_model))
            model_MD = np.array(model_MD)
            return model_MD

        def drop_outliers_by_overall_MD(self, percentile_thred=0.9973):
            """dropping outliers from the dataset
            """
            data = self.dataset[self.analog_tags_pid].values
            m = data.mean(axis=0)
            cov = 0
            for p in data:
                cov += np.dot((p - m).reshape(len(p), 1), (p - m).reshape(1, len(p)))
            cov /= (len(data) - 1)
            model_MD = cal_model_MD(data, m, cov)
            model_MD = pd.DataFrame(data=model_MD.T, columns=['md'], index=self.dataset.index)
            md_thred = model_MD.quantile([percentile_thred]).values[0][0]
            model_MD.loc[model_MD["md"] > md_thred, 'md'] = np.NaN
            model_MD.dropna(inplace=True)
            outliers_idx = self.dataset[~self.dataset.index.isin(model_MD.index)].index
            self.dataset = self.dataset[self.dataset.index.isin(model_MD.index)]
            return self.dataset, outliers_idx

        def cal_sensor_MD(self, y_error, mean_per_sensor, cov_per_sensor):
            """calculate Mahalanobis distance by sensor
            """
            sensor_MD = []
            for i in range(y_error.shape[1]):
                m = mean_per_sensor[i]
                c = cov_per_sensor[i]
                es = y_error[:, i]
                MD = []
                for e in es:
                    MD.append(self.Mahala_distantce(e, m, c))
                sensor_MD.append(MD)
            sensor_MD = np.array(sensor_MD)
            return sensor_MD

        def drop_outliers_by_mean_MD(self, percentile_thred=0.997):
            """dropping outliers by mean MD
            """
            try:
                data = self.dataset[self.analog_tags_pid].values
                mean_per_sensor = []
                cov_per_sensor = []
                for i in range(data.shape[1]):
                    err = np.reshape(data[:, i], (data.shape[0], -1))
                    mean = sum(err) / len(err)
                    mean_per_sensor.append(mean)
                    cov = 0
                    for e in err:
                        cov += np.dot((e - mean).reshape(len(e), 1), (e - mean).reshape(1, len(e)))
                    cov /= len(data) - 1
                    cov_per_sensor.append(cov)
                model_MD_mean = self.cal_sensor_MD(data, mean_per_sensor, cov_per_sensor).mean(axis=0)
                model_MD_mean = pd.DataFrame(data=model_MD_mean.T, columns=["MD"], index=self.dataset.index)
                md_threds = model_MD_mean.quantile([percentile_thred]).values[0]
                for i, pid in enumerate(model_MD_mean.columns):
                    model_MD_mean.loc[model_MD_mean[pid] > md_threds[i], pid] = np.NaN
                model_MD_mean.dropna(how='all', inplace=True)
                outliers_idx = self.dataset[~self.dataset.index.isin(model_MD_mean.index)].index
                self.dataset = self.dataset[self.dataset.index.isin(model_MD_mean.index)]
                return self.dataset, outliers_idx
            except Exception as e:
                print(e)
                print(analog_tags_pid[i])

        def drop_outliers_by_PHY(self, drop_thred=0.07, alert_thred=0.05):
            """dropping outliers by physical thresholds
            """
            outliers_info = {}
            physical_thresholds_alert = False
            try:
                for i, col in enumerate(self.dataset.columns):
                    if col not in self.analog_tags_pid:
                        continue
                    max_phy_thred = self.physical_thresholds[0][i]
                    min_phy_thred = self.physical_thresholds[1][i]
                    threshold_range = max_phy_thred - min_phy_thred
                    max_phy_thred_ratio = max_phy_thred + threshold_range * drop_thred
                    n_outliers = len(self.dataset.loc[self.dataset[col].values > max_phy_thred_ratio, col])
                    if n_outliers > 0:
                        ratio_max = n_outliers / len(self.dataset)
                    else:
                        ratio_max = 0.0
                    min_phy_thred_ratio = min_phy_thred - threshold_range * drop_thred
                    n_outliers = len(self.dataset.loc[self.dataset[col].values < min_phy_thred_ratio, col])
                    if n_outliers > 0:
                        ratio_min = n_outliers / len(self.dataset)
                    else:
                        ratio_min = 0.0
                    if ratio_max > 0.0 or ratio_min > 0.0:
                        outliers_info[col] = [
                            max_phy_thred,
                            max_phy_thred_ratio,
                            self.dataset[col].max(),
                            100 * ratio_max,
                            min_phy_thred,
                            min_phy_thred_ratio,
                            self.dataset[col].min(),
                            100 * ratio_min
                        ]
                    # need to modify physical thresholds if more than 5% data is out of physical range
                    if ratio_max > alert_thred or ratio_min > alert_thred:
                        physical_thresholds_alert = True

                    self.dataset.loc[self.dataset[col] > max_phy_thred_ratio, col] = np.NaN
                    self.dataset.loc[self.dataset[col] < min_phy_thred_ratio, col] = np.NaN

                outliers_idx = self.dataset[self.dataset.isna().any(axis=1)].index
                self.dataset.dropna(inplace=True)

            except Exception as e:
                print(e, col)

            return self.dataset, outliers_idx, outliers_info, physical_thresholds_alert

        def normalization(self, df, feature_range=(-1, 1)):
            """
            MaxMin normalization
            @parameters
            data          : np.array
            feature_range : tuple
                range of nomalization, default value is `(-1, 1)`
            @return
            scaler: scaler
                a new scaler of normalization
            std_data: np.array
                data after normalization
            """
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler = scaler.fit(df.values)
            std_data = scaler.transform(df.values)
            return scaler, std_data

        def drop_outliers(self):
            """interface for dropping outliers
            """
            try:
                if self.outlier_kind == 'MD':
                    # self.logger.info('(3) Drop outliers out of model overall MD')
                    self.dataset, _ = self.drop_outliers_by_overall_MD(self.dataset, self.analog_tags_pid)
                    return self.dataset, _
                elif self.outlier_kind == 'MD_mean':
                    # self.logger.info('(3) Drop outliers out of model mean MD')
                    self.dataset, _ = self.drop_outliers_by_mean_MD()
                    return self.dataset, _
                else:
                    # self.logger.info("(3) Drop outliers out of tolerant thresholds")
                    self.dataset, _, self.outliers_info, self.physical_thresholds_alert = self.drop_outliers_by_PHY(
                        self.dataset, self.analog_tags_pid, self.physical_thresholds)
                    return self.dataset, _
            except Exception as e:
                print(e)
                raise

    def create_subseq(ts, look_back=3, pred_length=1, interval=1):
            """
            Split data to X train data (look_back) and y prediction data (pred_length)
            @parameters
            ts          : np.array
                input data
            look_back   : int
                length of looking back
            pred_length : int
                length of prediction
            interval    : int
                frequency of splitting data, default is 1
            @return
            sub_seq    : 2D list
                X train data (look_back)
            next_values: 2D list
                y prediction data (pred_length)
            """
            sub_seq, next_values = [], []
            for i in range(0, len(ts) - look_back - pred_length + 1, interval):
                sub_seq.append(ts[i:i + look_back, :])
                next_values.append(ts[i + look_back:i + look_back + pred_length, :])
            return sub_seq, next_values


    def preprocess_data(outlier_kind='MD_mean', look_back=3, pred_length=1):
        """main function for prprocessing, runs all nessessary functions
        """
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        
        # getting previously collected dataset
        df = pd.read_csv(tar_data.path, index_col='time')
        df.index = pd.to_datetime(df.index)

        # getting tags data
        tags = pd.read_csv(tags_data.path)

        #processing tags data
        outlier_kind = outlier_kind  # 'MD_mean'
        analog_tags = tags[tags['type'] == 'A']
        digital_tags = tags[tags['type'] == 'D']
        digital_tags_pid = list(digital_tags['pid'].unique())
        analog_tags_pid = list(analog_tags['pid'].unique())
        analog_max_dict = dict(analog_tags[["pid", "max"]].values)
        analog_max_physical = [analog_max_dict[pid] for pid in analog_tags_pid]
        analog_min_dict = dict(analog_tags[["pid", "min"]].values)
        analog_min_physical = [analog_min_dict[pid] for pid in analog_tags_pid]
        n_digitals = len(digital_tags_pid)
        physical_thresholds = []
        physical_thresholds.append(analog_max_physical + [1] * n_digitals)
        physical_thresholds.append(analog_min_physical + [0] * n_digitals)

        if set(df.columns).issubset(analog_tags_pid):
            analog_tags_pid = list(df.columns)
        else:
            not_analog = list(set(df.columns) - set(analog_tags_pid))
            analog_tags_pid = [x for x in list(df.columns) if x not in not_analog]

        if set(df.columns).issubset(digital_tags_pid):
            digital_tags_pid = list(df.columns)
        else:
            not_digital = list(set(df.columns) - set(digital_tags_pid))
            digital_tags_pid = [x for x in list(df.columns) if x not in not_digital]

        tag_pid = analog_tags_pid + digital_tags_pid


        dataset = PreprocessData(df, outlier_kind, tag_pid, digital_tags_pid, physical_thresholds, n_digitals)
        df = dataset.get_dataset()
        df = dataset.drop_string()
        df = dataset.drop_error()

        try:
            df, _ = dataset.drop_outliers()
        except Exception as e:
            print(e)
        mapping_pid_to_name = dict(analog_tags[["pid", "pid_name_jp"]].values)
        mapping_pid_to_name.update(dict(digital_tags[["pid", "pid_name_jp"]].values))
        for i, pid in enumerate(df.columns):
            if pid in analog_tags_pid:
                print(
                    "{:>3d}/{} | A | {} | {:.3f} | {:.3f} | {}".format(
                        i + 1,
                        len(df.columns),
                        pid,
                        float(df[pid].min(skipna=True)),
                        float(df[pid].max(skipna=True)),
                        mapping_pid_to_name[pid]
                    )
                )
            elif pid in digital_tags_pid:
                print(
                    "{:>3d}/{} | D | {} | {} | {} | {}".format(
                        i + 1,
                        len(df.columns),
                        pid,
                        int(df[pid].min(skipna=True)),
                        int(df[pid].max(skipna=True)),
                        mapping_pid_to_name[pid]
                    )
                )
        scaler, scaled_data = dataset.normalization(df)

        df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        with open(config_art.path, 'rb') as f:
            config = pickle.load(f)
        output_tags_pid = config['output_pid_list']
        logging.info(f"output: {output_tags_pid}")
        output_tags_idx = [i for i, pid in enumerate(tag_pid) if pid in output_tags_pid]
        value_ranges = []
        for idx in output_tags_idx:
            value_ranges.append(scaler.data_max_[idx] - scaler.data_min_[idx])

        sub_seq = []
        next_values = []
        starttime = df.index[0]

        for i, val in enumerate(df.index):
            if i==len(df)-1:# or df.index[i] + timedelta(minutes=1) != df.index[i+1]:
                endtime=df.index[i]
                if starttime + timedelta(minutes=look_back+pred_length-1) <= endtime:
                    tmp=df[(df.index>=starttime)&(df.index<=endtime)]
                    tmp_sub_seq, tmp_next_values = create_subseq(tmp.values, look_back, pred_length)
                    if len(sub_seq) == 0:
                        sub_seq     = tmp_sub_seq
                        next_values = tmp_next_values
                    else:
                        sub_seq     = sub_seq + tmp_sub_seq
                        next_values = next_values + tmp_next_values
                if i<len(df)-1:
                    starttime = df.index[i+1]

        X_train, X_test, y_train, y_test = train_test_split(sub_seq, next_values, test_size=0.2, shuffle=False)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        resource = boto3.resource('s3',
                              aws_access_key_id = AWS_ACCESS_KEY_ID,
                              aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
        BUCKET = "pav-kf-test"
        with open(X_train_art.path, 'wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        resource.Bucket(BUCKET).upload_file(X_train_art.path, "data/X_test.npy")
        with open(X_test_art.path, 'wb') as handle:
            pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        resource.Bucket(BUCKET).upload_file(X_test_art.path, "data/X_train.npy")
        with open(y_train_art.path, 'wb') as handle:
            pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        resource.Bucket(BUCKET).upload_file(y_train_art.path, "data/y_test.npy")
        with open(y_test_art.path, 'wb') as handle:
            pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        resource.Bucket(BUCKET).upload_file(y_test_art.path, "data/y_train.npy")
        with open(scaler_art.path, 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_tags_pid_art.path, 'wb') as handle:
            pickle.dump(output_tags_pid, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_tags_idx_art.path, 'wb') as handle:
            pickle.dump(output_tags_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(mapping_pid_to_name_art.path, 'wb') as handle:
            pickle.dump(mapping_pid_to_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metrics.log_metric('X_train.shape', str(X_train.shape))
        metrics.log_metric('y_train.shape', str(y_train.shape))
        metrics.log_metric('X_test.shape', str(X_test.shape))
        metrics.log_metric('y_test.shape', str(y_test.shape))


    logging.debug("Starting preprocessing.")
    preprocess_data()
