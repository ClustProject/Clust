import time
import pandas as pd
from dateutil.parser import parse
from datetime import datetime, timedelta
pd.set_option('mode.chained_assignment',  None)
import re,logging

class CleanDataByType():
    def __init__(self, data, selected_columns, time_column, dtcpm, field_type):
        """
        Data Type 별 중복이 없이 하나의 시간 스탬프를 갖는 테이블 데이터 형태로 정리해주는 Class
        :param data : 저장을 위해 정리하고 싶은 데이터
        :type : DataFrame
        
        :param selected_columns : 저장하고 싶은 columns만 선택해 기입한 parameter, 변경하고 싶은 columns의 이름도 기입 가능
        :type : list
        
        :param time_column : 데이터 저장할 시 시간 스탬프로 지정할 column 이름, 여러개의 시간 column이 존재할 시(type 4) 여러개를 기입하여 시간 스탬프를 병합한 후 하나의 단일 시간 스탬프로 변환
        :type : string or dictionary
        :param dtcpm : duplicated time column processing method Parameter 로 시간이 중복되는 데이터 타입(type 3)인 경우 중복된 시간에 해당하는 value 를 처리하는 방법에 대한 parameter
        :type : list
        >>> selected_columns = [{"Selected_columns":['out_pm25_보정전', 'out_pm10_보정전', 'out_온도', "out_습도"]},
                                {"Rename_columns":["out_pm25_raw", "out_pm10_raw", "out_temp", "out_humi"]}]
        >>> time_column = {"Year":"거래일", "Month":"거래일", "Day":"거래일", "Hour":"시간대", "Minute":"-", "Second":"-"}
        >>> dtcpm = [{"Selected_columns":["out_humi", "out_temp"]}, 
                    {"Processing_method":["Max", "Min"]}]
        :return self.data : data - 분석 및 활용이 용이하도록 유일한 시간 스탬프를 갖는 테이블 데이터 형태로 정의
        :type : DataFrame
        
        """
        self.data = data
        self.selected_columns = selected_columns
        self.time_column = time_column
        self.dtcpm = dtcpm
        self.field_type = field_type

    ## 시간 Column 병합 및 24시 변환 (Data Type 4)
    def time_combine_conversion(self):
        if self.time_column["Year"] != "-":
            if self.time_column["Year"] == self.time_column["Month"] == self.time_column["Day"]:
                self.data["date"] = pd.to_datetime(self.data[self.time_column["Year"]])
            else:
                date_list = pd.unique(list(x for x in (self.time_column["Year"], self.time_column["Month"], self.time_column["Day"]) if x != "-")).tolist()
                self.data["date"] = self.data[date_list[0]].astype(str)

                if self.time_column["Day"]=="-":
                    for column in range(1,len(date_list)):
                        self.data["date"] = self.data["date"] + "-" + self.data[date_list[column]].astype(str)+"-01"
                else:
                    for column in range(1,len(date_list)):
                        self.data["date"] = self.data["date"] + "-" + self.data[date_list[column]].astype(str)
                self.data["date"] = pd.to_datetime(self.data["date"], format="%y-%m-%d")
        else:
            now_time = datetime.now()
            self.data["date"] = pd.to_datetime(now_time.date())

        if self.time_column["Hour"] != "-":
            if self.time_column["Hour"] == self.time_column["Minute"] == self.time_column["Second"]:
                self.data["time"] = pd.to_datetime(self.data[self.time_column["Hour"]])            
            else:
                time_list = pd.unique(list(x for x in (self.time_column["Hour"], self.time_column["Minute"], self.time_column["Second"]) if x != "-")).tolist()
                self.data["time"] = self.data[time_list[0]].astype(str)

                if self.time_column["Minute"] == "-":
                        self.data["time"]=self.data["time"] + ":00:00"
                elif self.time_column["Second"] == "-":
                    for column in range(1,len(time_list)):
                        self.data["time"] = self.data["time"] + ":" + self.data[time_list[column]].astype(str) + ":00"
                else:
                    for column in range(1,len(time_list)):
                        self.data["time"] = self.data["time"] + ":" + self.data[time_list[column]].astype(str)
                self.data["time"] = pd.to_datetime(self.data["time"], format="%H:%M:%S").dt.time
        else:
            self.data["time"] = pd.to_datetime("00:00:00", format="%H:%M:%S").dt.time

        self.data["time"]=list(map(lambda x,y : datetime.combine(x,y), self.data["date"], self.data["time"]))

        self.time_column = "time"
        self.data.set_index(self.data["time"], inplace=True)
        


    def time_combine_conversion_24(self):
        try:
            if self.time_column["Year"] != "-":
                if self.time_column["Year"] == self.time_column["Month"] == self.time_column["Day"]:
                    self.data["date"] = pd.to_datetime(self.data[self.time_column["Year"]])
                else:
                    date_list = pd.unique(list(x for x in (self.time_column["Year"], self.time_column["Month"], self.time_column["Day"]) if x != "-")).tolist()
                    self.data["date"] = self.data[date_list[0]].astype(str)

                    if self.time_column["Day"]=="-":
                        for column in range(1,len(date_list)):
                            self.data["date"] = self.data["date"] + "-" + self.data[date_list[column]].astype(str)+"-01"
                    else:
                        for column in range(1,len(date_list)):
                            self.data["date"] = self.data["date"] + "-" + self.data[date_list[column]].astype(str)
                    self.data["date"] = pd.to_datetime(self.data["date"], format="%y-%m-%d")
            else:
                now_time = datetime.now()
                self.data["date"] = pd.to_datetime(now_time.date())

            # 24hours conversion 00hour
            idx_24=[]
            data_time = self.data[[self.time_column["Hour"]]].astype(str)
            for idx, row in data_time.iterrows():
                if self.time_column["Minute"] == self.time_column["Hour"]:
                    if row[0][:2] == "24":
                        row[0] = "00"+row[0][2:]
                        idx_24.append(idx)
                else:
                    if row[0] =="24":
                        row[0] = "00"
                        idx_24.append(idx)
            self.data[self.time_column["Hour"]] = data_time[self.time_column["Hour"]]
            
            if self.time_column["Hour"] != "-":
                if self.time_column["Hour"] == self.time_column["Minute"] == self.time_column["Second"]:
                    self.data["time"] = pd.to_datetime(self.data[self.time_column["Hour"]])            
                else:
                    time_list = pd.unique(list(x for x in (self.time_column["Hour"], self.time_column["Minute"], self.time_column["Second"]) if x != "-")).tolist()
                    self.data["time"] = self.data[time_list[0]].astype(str)

                    if self.time_column["Minute"] == "-":
                            self.data["time"]=self.data["time"] + ":00:00"
                    elif self.time_column["Second"] == "-":
                        for column in range(1,len(time_list)):
                            self.data["time"] = self.data["time"] + ":" + self.data[time_list[column]].astype(str) + ":00"
                    else:
                        for column in range(1,len(time_list)):
                            self.data["time"] = self.data["time"] + ":" + self.data[time_list[column]].astype(str)
                    self.data["time"] = pd.to_datetime(self.data["time"], format="%H:%M:%S").dt.time
            else:
                self.data["time"] = pd.to_datetime("00:00:00", format="%H:%M:%S").dt.time

            self.data["date"].loc[idx_24] = self.data["date"].loc[idx_24] + timedelta(hours=24)
            self.data["time"]=list(map(lambda x,y : datetime.combine(x,y), self.data["date"], self.data["time"]))

            self.time_column = "time"
            self.data.set_index(self.data["time"], inplace=True)
        except:
                self.time_parse()

    def time_conversion_24(self):
        self.data.reset_index(inplace=True)
        try:
            idx_24=[]
            data_time = self.data[[self.time_column]].astype(str)

            if len(re.findall('[\W]', data_time[self.time_column][0])) == 0:
                for idx, row in data_time.iterrows():
                    if row[0][8:10] == "24":
                        row[0] = row[0][:8]+"00"+row[0][10:]
                        idx_24.append(idx)
            else:
                if (len(re.findall('[\W]', data_time[self.time_column][0][:8])) == 0) & (data_time[self.time_column][0][8]==" "):
                    for idx, row in data_time.iterrows():
                        if row[0][9:11] == "24":
                            if len(row[0]) == 11:
                                row[0] = row[0][:9]+"00"
                                idx_24.append(idx)
                            else:
                                row[0] = row[0][:9]+"00"+row[0][11:]
                                idx_24.append(idx)
                elif (len(re.findall('[\W]', data_time[self.time_column][0][:10])) == 2) & (data_time[self.time_column][0][10]==" "):
                    for idx, row in data_time.iterrows():
                        if row[0][11:13] == "24":
                            if len(row[0]) == 13:
                                row[0] = row[0][:11]+"00"
                                idx_24.append(idx)
                            else:
                                row[0] = row[0][:11]+"00"+row[0][13:]
                                idx_24.append(idx)
            
            data_time[self.time_column] = pd.to_datetime(data_time[self.time_column])
            data_time.loc[idx_24] = data_time.loc[idx_24] + timedelta(hours=24)
            self.data[self.time_column] = data_time

            self.data.set_index(self.data[self.time_column], inplace=True)
        except :
                self.time_parse()

    def time_parse(self):
        try:
            self.data[self.time_column] = parse(self.data[self.time_column])
            self.data.set_index(self.time_column, inplace=True)
        except:
            print("Time Column Error")
    
    ## 시간 중복 시 데이터 처리하는 부분 (Data Type 3)
    def duplicate_column(self):   
        duplicate_dict ={}
        for n in range(len(self.dtcpm[0]["Selected_columns"])):
            if self.dtcpm[1]["Processing_method"][n] == "Remove":
                duplicate_dict[self.dtcpm[0]["Selected_columns"][n]] = list(self.data.loc[~self.data.index.duplicated(keep="first")][self.dtcpm[0]["Selected_columns"][n]])
            elif self.dtcpm[1]["Processing_method"][n] == "Sum":
                duplicate_dict[self.dtcpm[0]["Selected_columns"][n]] = list(self.data[self.dtcpm[0]["Selected_columns"][n]].groupby("time").sum())
            elif self.dtcpm[1]["Processing_method"][n] == "Average":
                duplicate_dict[self.dtcpm[0]["Selected_columns"][n]] = list(self.data[self.dtcpm[0]["Selected_columns"][n]].groupby("time").mean())
            elif self.dtcpm[1]["Processing_method"][n] == "Max":
                duplicate_dict[self.dtcpm[0]["Selected_columns"][n]] = list(self.data[self.dtcpm[0]["Selected_columns"][n]].groupby("time").max())
            elif self.dtcpm[1]["Processing_method"][n] == "Min":
                duplicate_dict[self.dtcpm[0]["Selected_columns"][n]] = list(self.data[self.dtcpm[0]["Selected_columns"][n]].groupby("time").min())

        data_duplicate = pd.DataFrame(duplicate_dict)
        data_duplicate.set_index(self.data.loc[~self.data.index.duplicated(keep="first")].index, inplace = True)

        return data_duplicate

    ## Data Clean
    def clean_data(self):
        if type(self.time_column) == dict:
            time_list = pd.unique(list(x for x in (self.time_column.values()) if x != "-")).tolist()
            for num in range(len(time_list)):
                self.data = self.data.dropna(subset=[time_list[num]], axis=0)
            try:
                self.time_combine_conversion()
            except ValueError:
                self.time_combine_conversion_24()
        else:
            self.data = self.data.dropna(subset=[self.time_column], axis=0)
            try:
                self.data.set_index(self.time_column, inplace=True)
                self.data.index = pd.to_datetime(self.data.index)
            except ValueError:
                self.time_conversion_24()

        if type(self.selected_columns[0]) == dict:
            columns_dict={}
            for n in range(len(list(self.selected_columns[0].values())[0])):
                columns_dict[self.selected_columns[0]["Selected_columns"][n]] = self.selected_columns[1]["Rename_columns"][n]
            self.data = self.data.rename(columns=columns_dict)
            self.data = self.data[self.selected_columns[1]["Rename_columns"]]
        else:
            self.data = self.data[self.selected_columns]
        self.data.index.names = ["time"]
        if self.dtcpm != None:
            self.data = self.duplicate_column()
        if self.field_type:
            self.data = self.data.astype(self.field_type)

        return self.data