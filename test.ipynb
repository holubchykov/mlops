%%writefile load_data.py
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
    packages_to_install=['boto3', 'pandas'],
)
def load_data(platfrom: str, periods: list,  output_csv: Output[Dataset], tags_csv: Output[Dataset], metrics: Output[Metrics]):

    from datetime import timedelta

    import logging
    import boto3
    from dataclasses import dataclass, field
    from typing import List
    from datetime import datetime

    import pandas as pd
    from io import StringIO
    AWS_ACCESS_KEY_ID="AKIAZTN6P4WBRRX2YXAG"
    AWS_SECRET_ACCESS_KEY="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"
    @dataclass
    class DataHelper():
        """
        Data class with the parameters
        """
        s3_bucket: str = "jcc-ai-lab-dev-jaime"
        plant: str = 'hk'
        unit: str = '01'
        pid_type: str = field(default_factory=lambda: ['a','d'])
        pid_list: List = field(default_factory=lambda: ["HK_1C7025","HK_1W7000","HK_DK0010","HK_1A2007","HK_1A2010"])
        output_pid_list: List = field(default_factory=lambda: ["HK_1W7000","HK_DK0010"])
        train_periods: List = field(default_factory=lambda: periods)
        prefix = 'DLZ/'
        table_pre = 'dl_t'
        model = "anomaly-detection"
        version = "1"

    class LoadData():
        """
        Collect data from storage,
        working as a universal interface for different types of storages
        @parameters
            platform: string
                name of the platform
            data_helper: dataclass
                instance of dataclass with the parameters
        """

        def __init__(self, platform, data_helper) -> None:
            self.parameters = data_helper
            if platform=="s3":
                self.platform = FromS3(self.parameters)
            elif platform=="azure":
                self.platform = FromAzure()
            elif platform=="kubernetes":
                self.platform = FromKubernetes(self.parameters, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            else:
                logging.error(f"Platform {platform} is not supported, choose one platform type from [s3, azure, kubernetes]")

        def get_data(self):
            """General method to get data
            @return
            df: pd.DataFrame
                IoT data
            """
            logging.info('starting getting data')
            time_ = datetime.now()
            self.platform.get_files()
            self.data = self.platform.get_data()
            self.data['time']= pd.to_datetime(self.data['time'])
            self.data.drop('quality', axis=1, inplace=True)
            #print(self.data)
            logging.info('returning dataframe')
            logging.info(f'time taken to collect dataframe: {datetime.now() - time_}')
            return self.data

        def pivot_data(self):
            """General method to pivot data
                and set index as a timeframe
            @return
            df: pd.DataFrame
                IoT data
            """
            logging.info('starting to pivot data')
            time_ = datetime.now()
            self.data.drop_duplicates(subset=['time', 'pid_no'], keep='first', inplace=True)
            self.data = self.data.pivot(index='time', columns='pid_no', values='value')
            logging.info('finished data pivot')
            logging.info(f'time taken to pivot dataframe: {datetime.now() - time_}')
            return self.data


        def save_data(self):
            """General method to save data to s3
            @return
            path: string
                local path to collected dataset
            """
            logging.info(f'Starting to save data: {output_csv.path}')
            self.data.to_csv(output_csv.path)
            logging.info(f'Finished saving data: {output_csv.path}')

            logging.info(f'starting to save tags data locally')
            client = boto3.client('s3',
                                      aws_access_key_id = AWS_ACCESS_KEY_ID,
                                      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
            csv_obj = client.get_object(Bucket='jcc-ai-lab-dev-jaime', Key='DLZ/dl_m_pid/2022/10/06/dl_m_pid_F20221006.csv')
            body = csv_obj['Body']
            csv_string = body.read().decode('utf-8')
            tags = pd.read_csv(StringIO(csv_string), low_memory=False)
            tags.to_csv(tags_csv.path)
            logging.info('finished saving tags data')

            
    class FromKubernetes():
        def __init__(self, parameters, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) -> None:
            self.client =boto3.client('s3',
                                      aws_access_key_id = AWS_ACCESS_KEY_ID,
                                      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
            self.resource = boto3.resource('s3',
                                           aws_access_key_id = AWS_ACCESS_KEY_ID,
                                           aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
            self.files_list = []
            self.parameters = parameters


        def get_files(self)->None:
            """
            creating list of csv files that need to be collected 
            """
            logging.info('starting to collect file objects')
            time_ = datetime.now()
            for pid_type in self.parameters.pid_type:
                tb_name = '_'.join([self.parameters.table_pre, self.parameters.plant, self.parameters.unit, pid_type])
                for period in self.parameters.train_periods:
                    start_dt = datetime.strptime(period[0], "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(period[1], "%Y-%m-%d %H:%M:%S")
                    n_days = (end_dt - start_dt).days
                    while start_dt <= end_dt:
                        year = f'{start_dt.year}'
                        month = f'{start_dt.month:02d}'
                        day = f'{start_dt.day:02d}'
                        prefix = f"{self.parameters.prefix}{tb_name}/{year}/{month}/{day}/"    
                        results = self.client.list_objects(Bucket=self.parameters.s3_bucket, Prefix=prefix, Delimiter='/')
                        #print(results)
                        start_dt = start_dt + timedelta(days=1)
                        if 'Contents' not in results:
                            continue
                        for file in results['Contents']:
                            self.files_list.append(file['Key'])      
            logging.info('finished to collect file objects')
            logging.info(f'time taken to collect file objects: {datetime.now() - time_}')        


        def get_data(self):
            """iterating over the list of collected files
            @return
            df: pd.DataFrame
                raw dataset
            """
            logging.info('starting to collect csv files')
            time_ = datetime.now()
            full_df = pd.DataFrame()
            for data_key in self.files_list:
                csv_obj = self.client.get_object(Bucket=self.parameters.s3_bucket, Key=data_key)
                body = csv_obj['Body']
                csv_string = body.read().decode('utf-8')
                raw_df = pd.read_csv(StringIO(csv_string), low_memory=False)
                if not raw_df.empty:
                    raw_df1 = pd.DataFrame()
                    raw_df = raw_df.loc[raw_df['quality'] == 0]
                    raw_df = raw_df[raw_df['pid_no'].isin(self.parameters.pid_list)]
                    raw_df['time']= pd.to_datetime(raw_df['time'])
                    raw_df.index = raw_df['time']
                    for period in self.parameters.train_periods:
                        t1 = datetime.strptime(period[0],'%Y-%m-%d %H:%M:%S').time().isoformat()
                        t2 = datetime.strptime(period[1],'%Y-%m-%d %H:%M:%S').time().isoformat()
                        if raw_df1.empty:
                            raw_df1 = raw_df.between_time(t1,t2)
                        else:
                            raw_df1 = raw_df1.append(raw_df.between_time(t1,t2), ignore_index = True)
                    if full_df.empty:
                        full_df = raw_df1
                    else:
                        full_df = full_df.append(raw_df1, ignore_index = True)
            logging.info('finished to collect csv files')
            logging.info(f'time taken to collect csv files: {datetime.now() - time_}')
            return full_df

    logging.info("start to load raw dataset")
    data_loader = LoadData('kubernetes', DataHelper())
    df = data_loader.get_data()
    df = data_loader.pivot_data()
    data_loader.save_data()
    metrics.log_metric('len_df', len(df))
