from gc import collect
import time
from datetime import datetime, timedelta
import pandas as pd
from dateutil.parser import parse
pd.set_option('mode.chained_assignment',  None)
import re,logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Clust.clust.ingestion.CSVtoInflux.cleanDataByType import CleanDataByType as CDBT

class Collector():
    def __init__(self, collect_parameter, db_client):
        """
        데이터 수집 및 저장하는 Class

        :param collect_parameter : 데이터를 저장하기에 필요한 Parameter
        :type collect_parameter : dictionary
        
        :param db_client: instance to get data from influx DB
        :type db_client: instance of influxClient class
        
        >>> collect_parameter = {
                "uploadType" : upload_type,
                "dataReadType" : data_read_type,
                "dbName" : db_name,
                "msName" : { 
                    "folder_fileListName_flag" : True, # folder안의 file list 이름으로 msName을 지정할 시 True (Folder 업로드시 사용)
                    "name" : None # folder_fileListName_flag가 True 인 경우 직접 msName 을 기입하는 name 은 None 기입
                },
                "filePath" : path,
                "selectedDatas" : selected_datas, # data read type 이 selectedData 인 경우 필요한 param
                "timeColumnName" : time_column,
                "selectedColumns" : selected_columns,
                "duplicatedTimeColumnProcessingMethod" : dtcpm,
                "encoding" : encoding
            }
        """
        ## InfluxDB
        self.db_client = db_client
        
        ## Type 별 Upload, Data Read
        self.upload_type = collect_parameter["uploadType"]
        self.data_read_type = collect_parameter["dataReadType"]
        
        ## 기본 Data 정보
        self.db_name = collect_parameter["dbName"]
        self.ms_name = collect_parameter["msName"]
        
        ## Data Read 시 필요한 정보
        self.file_path = collect_parameter["filePath"]
        self.encoding = collect_parameter["encoding"]
        self.selected_datas = collect_parameter["selectedDatas"]
        
        ## Clean Data 시 필요한 정보
        self.time_column = collect_parameter["timeColumnName"]
        self.selected_columns = collect_parameter["selectedColumns"]
        self.dtcpm = collect_parameter["duplicatedTimeColumnProcessingMethod"]
        self.field_type = collect_parameter["fieldType"]
        
       ## 정한 parameter 랑 def 함수를 일단 여기다가 붙여보기

    def get_basic_data(self):
        if type(self.time_column) == dict:
            self.data = pd.read_csv(self.file_path, header=0, index_col=False, encoding=self.encoding, dtype={self.time_column["Year"]:str})
        else:
            self.data = pd.read_csv(self.file_path, header=0, index_col=False, encoding=self.encoding, dtype={self.time_column:str})
            
# self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] == self.selected_datas[1]["Selected_values"][n]].copy()
    def get_data_by_condition(self):
        self.get_basic_data()
        for n in range(len(self.selected_datas[0]["Selected_columns"])):
            if self.selected_datas[2]["Selected_Function"][n] == "Equal":
                self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] == self.selected_datas[1]["Selected_values"][n]].copy()
            elif self.selected_datas[2]["Selected_Function"][n] == "Above":
                self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] >= self.selected_datas[1]["Selected_values"][n]].copy()
            elif self.selected_datas[2]["Selected_Function"][n] == "Below":
                self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] <= self.selected_datas[1]["Selected_values"][n]].copy()
            elif self.selected_datas[2]["Selected_Function"][n] == "Exceeded":
                self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] > self.selected_datas[1]["Selected_values"][n]].copy()
            elif self.selected_datas[2]["Selected_Function"][n] == "Less than":
                self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] < self.selected_datas[1]["Selected_values"][n]].copy()
            elif self.selected_datas[2]["Selected_Function"][n] == "Exception":
                self.data = self.data[self.data[self.selected_datas[0]["Selected_columns"][n]] != self.selected_datas[1]["Selected_values"][n]].copy()

    def get_high_capacity_data(self):
        self.dataset = pd.read_csv(self.file_path, chunksize=25000, encoding=self.encoding, error_bad_lines=False)        
    
    def set_selected_column(self):
        if self.selected_columns == None:
            columns = list(self.data.columns)
            columns = [column for column in columns if column not in self.time_column]
            self.selected_columns = columns

    def write_data(self):
        print("Writing Data ...")
        print('=========== ms name : '+self.ms_name+' ===========')
        print(self.data.tail(2))
        
        self.db_client.write_db(self.db_name, self.ms_name, self.data)
        #db_client.write_db(self.db_name, self.ms_name, self.data)
        # bk_name, ms_name, data_frame

    def collect_clean_data(self, data):
        """
        data를 clean, write 하는 함수
        param data: 사용자가 저장하고 싶은 데이터
        type data: dataframe
        """
        cdbt = CDBT(data, self.selected_columns, self.time_column, self.dtcpm, self.field_type)
        self.data = cdbt.clean_data()
        print("===========data clean success===========")
        self.write_data()

    def collect_clean_dataset(self):
        """
        dataset을 clean, write 하는 함수로 대용량 데이터일 경우 이 함수를 활용
        """
        for data in self.dataset:
            self.collect_clean_data(data)

    def collect_by_data_read_type(self):
        """
        data_read_type(basic, selectedData, highCapacity)에 따라 데이터를 읽고 저장하는 함수
        """
        if self.data_read_type == "highCapacity":
            self.get_high_capacity_data()
            self.collect_clean_dataset()
        else:
            if self.data_read_type == "basic":
                self.get_basic_data()
            elif self.data_read_type == "selectedData":
                self.get_data_by_condition()
            self.collect_clean_data(self.data)
        
    
    def collect(self):
        if self.upload_type == "File":
            self.collect_by_data_read_type()
        elif self.upload_type == "Folder":
            count=1
            filenames = os.listdir(self.file_path)
            folder_path = self.file_path
            for filename in filenames:
                print("############# count : ", count, " #############")
                count+=1
                self.file_path = os.path.join(folder_path, filename)
                self.ms_name = filename.split(".")[0]
                self.collect_by_data_read_type()
        self.db_client.DBClient.close()