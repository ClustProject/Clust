from doctest import DocFileCase
import warnings
#from influxdb_client.client.warnings import MissingPivotFunction
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS, WriteOptions
from influxdb_client import InfluxDBClient, Point, BucketsService, Bucket
import sys
import os
import pandas as pd
from datetime import datetime

#warnings.simplefilter("ignore", MissingPivotFunction)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

UTC_Style = '%Y-%m-%dT%H:%M:%SZ'
class influxClient():
    """
    Influx DB 2.0 Connection

        **Standard Influx Query**::

            from(bucket:"bucket_name")
            |> range(start: start_time, stop: end_time)
            |> filter(fn: (r) => r._measurement == "measurement_name")

        **change result of Influx 2.0 to Influx 1.8**::

            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    def __init__(self, influx_setting):
        self.influx_setting = influx_setting
        self.DBClient = InfluxDBClient(url=self.influx_setting["url"], token=self.influx_setting["token"], org=self.influx_setting["org"], timeout=30000_000)



    def get_DBList(self):
        """
        get all bucket(Database) list

        :return: db_list
        :rtype: List
        """
        buckets_api = self.DBClient.buckets_api()
        buckets = buckets_api.find_buckets(limit=100).buckets  # bucket list 보여주기 최대 100까지만 가능

        bk_list = []
        bk_list.extend(bucket.name for bucket in buckets)

        bk_list = [bk for bk in bk_list if bk not in ['_monitoring', '_tasks', 'telegraf']]

        return bk_list

    def measurement_list(self, bk_name):
        """
        get all measurement list of specific Bucket

        :param bk_name: bucket(database) 
        :type bk_name: string

        :return: measurement list
        :rtype: List
        """
        query = f'import "influxdata/influxdb/schema" schema.measurements(bucket: "{bk_name}")'
        ms_list = []
        try:
            query_result = self.DBClient.query_api().query_data_frame(query)
            ms_list = list(query_result["_value"])
        except Exception as e:
            print(e)

        return ms_list

    def measurement_list_only_start_end(self, bk_name):
        """
        Get the only start and end measurement name
        Use this function to reduce the DB load time.

        :param db_name: bucket(database) 
        :type db_name: string
        :return: measurement list
        :rtype: List
        """
        ms_list = []
        ori_ms_list = self.measurement_list(bk_name)
        ori_len = len(ori_ms_list)

        if(ori_len == 1):
            ms_list.append(ori_ms_list[0])
        elif(ori_len == 2):
            ms_list.append(ori_ms_list[0])
            ms_list.append(ori_ms_list[len(ori_ms_list)-1])
        elif(ori_len > 2):
            ms_list.append(ori_ms_list[0])
            ms_list.append("...(+"+str(ori_len-2)+")")
            ms_list.append(ori_ms_list[len(ori_ms_list)-1])

        return ms_list

    def get_fieldList(self, bk_name, ms_name, onlyFieldName= False):
        """
        get all field list of specific measurements

        :param db_name: bucket(database) 
        :type db_name: string
        :param ms_name: measurement 
        :type ms_name: string

        :return: fieldList in measurement
        :rtype: List
        """
        query = f'''
        import "experimental/query"

        query.fromRange(bucket: "{bk_name}", start:0)
        |> query.filterMeasurement(
            measurement: "{ms_name}")
        |> keys()
        |> distinct(column: "_field")
        '''
        
        query_result = self.DBClient.query_api().query_data_frame(query=query)
        field_list = list(query_result["_field"])
        field_list = list(set(field_list))
        return field_list

    def get_data(self, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get :guilabel:`all data` of the specific mearuement, change dataframe
        
        :param db_name: bucket(database) 
        :type db_name: string
        :param ms_name: measurement 
        :type ms_name: string
        :param tag_key: tagkey (option)
        :type ms_name: string

        :param tag_value: tagValue (option)
        :type ms_name: string

        :return: df, measurement data
        :rtype: DataFrame
        """
        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: 0, stop: now()) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
        else:
            query = f'''
            from(bucket:"{bk_name}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
        query_client = self.DBClient.query_api()
        data_frame = query_client.query_data_frame(query)
        data_frame = self.cleanup_df(data_frame)

        return data_frame

        # first() - 테이블에서 첫번째 레코드 반환
    def get_first_time(self, bk_name, ms_name):
        """
        Get the :guilabel:`first data` of the specific mearuement

        :param db_name: bucket(database) 
        :type db_name: string
        :param ms_name: measurement
        :type ms_name: string

        :return: first time in data
        :return: pandas._libs.tslibs.timestamps.Timestamp
        """
        query = f'''from(bucket: "{bk_name}") 
        |> range(start: 0, stop: now()) 
        |> filter(fn: (r) => r._measurement == "{ms_name}")
        |> group(columns: ["_field"])
        |> first()
        |> drop(columns: ["_start", "_stop", "_measurement"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        query_result = self.DBClient.query_api().query_data_frame(query=query)
        query_first_time = sorted(query_result["_time"])
        first_time = query_first_time[0]

        return first_time

        # last() - 테이블에서 마지막 레코드 반환
    def get_last_time(self, bk_name, ms_name):
        """
        Get the :guilabel:`last data` of the specific mearuement

        :param db_name: bucket(database) 
        :type db_name: string
        :param ms_name: measurement 
        :type ms_name: string

        :return: last time in data
        :rtype: pandas._libs.tslibs.timestamps.Timestamp
        """
        query = f'''
        from(bucket: "{bk_name}") 
        |> range(start: 0, stop: now()) 
        |> filter(fn: (r) => r._measurement == "{ms_name}")
        |> last()
        |> drop(columns: ["_start", "_stop", "_measurement"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        query_result = self.DBClient.query_api().query_data_frame(query=query)
        query_last_time = sorted(query_result["_time"],reverse=True)
        last_time = query_last_time[0]

        return last_time

    def get_data_by_time(self, start_time, end_time, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get data of the specific measurement based on :guilabel:`start-end duration`
        *get_datafront_by_duration(self, start_time, end_time)*

        :param start_time: start time
        :type start_time: pandas._libs.tslibs.timestamps.Timestamp or string

        :param end_time: end time
        :type end_time: pandas._libs.tslibs.timestamps.Timestamp or string

        :param db_name: database name
        :type db_name: string

        :param ms_name: measurement name
        :type ms_name: string

        :param tag_key: tagkey (option)
        :type ms_name: string

        :param tag_value: tag_value (option)
        :type ms_name: string

        :return: df, time duration
        :rtype: DataFrame
        """
        if isinstance(start_time, str):
            if 'T' not in start_time:
                if len(start_time) < 12:
                    start_time = start_time + " 00:00:00"
                    end_time = end_time + " 23:59:59"
                start_time = datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
                end_time = datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
            else:
                pass
        else: #Not String:
            start_time = start_time.strftime(UTC_Style)
            end_time = end_time.strftime(UTC_Style)

        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: {start_time}, stop: {end_time}) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
        else:
            query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: {start_time}, stop: {end_time}) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
        data_frame = self.DBClient.query_api().query_data_frame(query=query)

        data_frame = self.cleanup_df(data_frame)

        return data_frame

    def get_data_by_days(self, end_time, days, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get data of the specific mearuement based on :guilabel:`time duration` (days)

        :param end_time: end time 
        :type end_time: pandas._libs.tslibs.timestamps.Timestamp

        :param days: duration days
        :type days: integer

        :param db_name: database
        :type db_name: string

        :param ms_name: measurement
        :type ms_name: string

        :return: df, time duration
        :rtype: DataFrame

        """
        if isinstance(end_time, str):
            if 'T' not in end_time:
                if len(end_time) < 12:
                    end_time = end_time + " 23:59:59"
                end_time = datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
            else:
                pass
        else: #Not String:
            end_time = end_time.strftime(UTC_Style)

        if tag_key:
            if tag_value:
                query = f'''
                import "experimental"
                from(bucket: "{bk_name}") 
                |> range(start: experimental.subDuration(d: {days}d, from: {end_time}), stop: now())
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
        else:
            query = f'''
            import "experimental"
            from(bucket: "{bk_name}") 
            |> range(start: experimental.subDuration(d: {days}d, from: {end_time}), stop: now())
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
        query_client = self.DBClient.query_api()
        data_frame = query_client.query_data_frame(query=query)
        data_frame = self.cleanup_df(data_frame)

        return data_frame

    def get_datafront_by_num(self, number, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get the :guilabel:`first N number` data from the specific measurement
        
        :param db_name: number(limit) 
        :type db_name: integer

        :param db_name: bucket(database)   
        :type db_name: string

        :param ms_name: measurement 
        :type ms_name: string

        :return: df, first N(number) row data in measurement
        :rtype: DataFrame
        """
        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: 0, stop: now()) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> limit(n:{number})
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
        else:
            query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: 0, stop: now()) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> limit(n:{number})
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")            
            '''

        data_frame = self.DBClient.query_api().query_data_frame(query=query)
        data_frame = self.cleanup_df(data_frame)

        return data_frame

    def get_dataend_by_num(self, number, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get the :guilabel:`last N number` data from the specific measurement

        :param db_name: number(limit) 
        :type db_name: integer

        :param db_name: bucket(database)  
        :type db_name: string

        :param ms_name: measurement 
        :type ms_name: string

        :return: df, last N(number) row data in measurement
        :rtype: DataFrame
        """
        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: 0, stop: now()) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> tail(n:{number})
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''

        else:
            query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: 0, stop: now()) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> tail(n:{number})
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")   
            '''

        data_frame = self.DBClient.query_api().query_data_frame(query=query)
        data_frame = self.cleanup_df(data_frame)

        return data_frame


    def cleanup_df(self, df):
        """
        Clean data, remove duplication, Sort, Set index (datetime)

        - Set index to datetime
        - Remove duplication
        - Sort ascending
        - Replace blank to Nan

        :param df: dataFrame
        :type df: dataFrame

        :return: df, data setting
        :rtype: DataFrame
        """
        import numpy as np
        if 'result' in df.columns:
            df = df.drop(['result', 'table'], axis=1)
            if '_time' in df.columns:
                df = df.set_index('_time')
            elif 'time' in df.columns:
                df = df.set_index('time')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')
            df.index.name ='time'
            df = df.groupby(df.index).first()
            df.index = pd.to_datetime(df.index)
            # index의 중복된 행 중 첫째행을 제외한 나머지 행 삭제
            df = df.sort_index(ascending=True)
            df.replace("", np.nan, inplace=True)
        else:
            pass
        return df


    def get_freq(self, bk_name, ms_name, tag_key=None, tag_value=None): 
        """
        :param db_name: bucket(database)  
        :type db_name: string
        :param ms_name: measurement
        :type ms_name: string

        :return: freq
        :rtype: Dict
        """
        if tag_key:
            if tag_value:
                data = self.get_datafront_by_num(10,bk_name, ms_name,tag_key, tag_value)
        else:
            data = self.get_datafront_by_num(10,bk_name, ms_name)
        from KETIPrePartialDataPreprocessing.data_refine.frequency import RefineFrequency
        result = str(RefineFrequency().get_frequencyWith3DataPoints(data))
        return result


    def get_data_limit_by_time(self, start_time, end_time, number, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get the :guilabel:`limit data` of the specific mearuement based on :guilabel:`time duration` (days)
        
        
        :param start_time: start time
        :type start_time: pandas._libs.tslibs.timestamps.Timestamp

        :param end_time: end time 
        :type end_time: pandas._libs.tslibs.timestamps.Timestamp

        :param db_name: number(limit) 
        :type db_name: integer

        :param db_name: bucket(database)  
        :type db_name: string

        :param ms_name: measurement 
        :type ms_name: string


        :return: df, time duration
        :rtype: DataFrame
        """
        if isinstance(start_time, str):
            if 'T' not in start_time:
                if len(start_time) < 12:
                    start_time = start_time + " 00:00:00"
                    end_time = end_time + " 23:59:59"
                start_time = datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
                end_time = datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
            else:
                pass
        else: #Not String:
            start_time = start_time.strftime(UTC_Style)
            end_time = end_time.strftime(UTC_Style)
        
        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: {start_time}, stop: {end_time}) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> limit(n:{number})
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''

        else:
            query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: {start_time}, stop: {end_time}) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> limit(n:{number})
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
        data_frame = self.DBClient.query_api().query_data_frame(query)
        data_frame = self.cleanup_df(data_frame)

        return data_frame


    def get_data_count(self, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get the :guilabel:`data count` from the specific measurement

        :param db_name: bucket(database)  
        :type db_name: string

        :param ms_name: measurement 
        :type ms_name: string

        :return: data count
        :rtype: integer
        """
        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: 0, stop: now()) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
                data_frame = self.DBClient.query_api().query_data_frame(query)
                data_count = len(data_frame)
        else:
            query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: 0, stop: now()) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> group(columns: ["_field"])
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> count()
            '''
            data_frame = self.DBClient.query_api().query_data_frame(query)
            data_count = int(data_frame["_value"][0])

        return data_count


    def get_data_by_time_count(self, start_time, end_time, bk_name, ms_name, tag_key=None, tag_value=None):
        """
        Get data of the specific measurement based on :guilabel:`start-end duration`
        *get_datafront_by_duration(self, start_time, end_time)*

        :param start_time: start time
        :type start_time: pandas._libs.tslibs.timestamps.Timestamp or string

        :param end_time: end time
        :type end_time: pandas._libs.tslibs.timestamps.Timestamp or string

        :param db_name: database name
        :type db_name: string

        :param ms_name: measurement name
        :type ms_name: string

        :param tag_key: tagkey (option)
        :type ms_name: string

        :param tag_value: tag_value (option)
        :type ms_name: string

        :return: df, time duration
        :rtype: DataFrame
        """
        if isinstance(start_time, str):
            if 'T' not in start_time:
                if len(start_time) < 12:
                    start_time = start_time + " 00:00:00"
                    end_time = end_time + " 23:59:59"
                start_time = datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
                end_time = datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S').strftime(UTC_Style)
            else:
                pass
        else: #Not String:
            start_time = start_time.strftime(UTC_Style)
            end_time = end_time.strftime(UTC_Style)

        if tag_key:
            if tag_value:
                query = f'''
                from(bucket: "{bk_name}") 
                |> range(start: {start_time}, stop: {end_time}) 
                |> filter(fn: (r) => r._measurement == "{ms_name}")
                |> filter(fn: (r) => r.{tag_key} == "{tag_value}")
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''
        else:
            query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: {start_time}, stop: {end_time}) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> group(columns: ["_field"])
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> count()
            '''
        data_frame = self.DBClient.query_api().query_data_frame(query)
        data_count = int(data_frame["_value"][0])

        return data_count


    def create_bucket(self, bk_name):  # write_db 수행 시, bucket 생성 필요
        """
        Create bucket to the influxdb
        """
        buckets_api = self.DBClient.buckets_api()
        buckets_api.create_bucket(bucket_name=bk_name)
        print("========== create bucket ==========")


    def write_db(self, bk_name, ms_name, data_frame):
        """
        Write data to the influxdb
        """
        write_client = self.DBClient.write_api(write_options=WriteOptions(batch_size=10000))
        if bk_name not in self.get_DBList():
            self.create_bucket(bk_name)

        write_client.write(bucket=bk_name, record=data_frame,data_frame_measurement_name=ms_name)
        print("========== write success ==========")
        import time
        time.sleep(2)
    

    def drop_measurement(self, bk_name, ms_name):
        """
        Drop Measurement
        """
        start_time = '1970-01-01T00:00:00Z'
        end_time = datetime.now().strftime(UTC_Style)
        delete_api = self.DBClient.delete_api()
        delete_api.delete(start_time, end_time, f'_measurement={ms_name}', bucket=bk_name, org = self.influx_setting["org"])
        print("========== drop measurement ==========")







# --------------------------------------------- new function from wiz ---------------------------------------------
    def get_tagList(self, bk_name, ms_name):

        query = f'''
            from(bucket: "{bk_name}") 
            |> range(start: 0, stop: now()) 
            |> filter(fn: (r) => r._measurement == "{ms_name}")
            |> limit(n:1)
            |> group(columns: ["_field"])
            |> drop(columns: ["_start", "_stop", "_measurement","_field","_value","_time"])
            '''
        query_result = self.DBClient.query_api().query_data_frame(query=query)
        tag_list = list(query_result.columns[2:])

        return tag_list


    def get_TagValue(self, bk_name, ms_name, tag_key):
        """
        Get :guilabel:`unique value` of selected tag key

        :param db_name: bucket(database) 
        :type db_name: string
f
        :param ms_name: measurement
        :type ms_name: string

        :param tag_key: select tag key data
        :type tag_key: string

        :return: unique tag value list
        :rtype: List
        """

        query = f'''
        import "experimental/query"

        query.fromRange(bucket: "{bk_name}", start:0)
        |> query.filterMeasurement(
            measurement: "{ms_name}")
        |> keys()
        |> distinct(column: "{tag_key}")
        '''
        query_result = self.DBClient.query_api().query_data_frame(query=query)
        query_result = query_result.drop_duplicates([tag_key])
        tag_value = list(query_result[tag_key])

        return tag_value


    def get_fieldList_type(self, bk_name, ms_name, onlyFieldName= False):
        query = f'''from(bucket: "{bk_name}") 
        |> range(start: 0, stop: now()) 
        |> filter(fn: (r) => r._measurement == "{ms_name}")
        |> group(columns: ["_field"])
        |> first()
        |> drop(columns: ["_start", "_stop", "_measurement"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        query_result = self.DBClient.query_api().query_data_frame(query=query)
        column_df = self.cleanup_df(query_result)

        field_list = []
        dtype_series = column_df.dtypes

        tag_list = self.get_tagList(bk_name, ms_name)

        for dtype_index, dtype_column in enumerate(dtype_series.index):
            dtype_dict = {}
            dtype_type = str(dtype_series.values[dtype_index])

            if dtype_type == 'object':
                dtype_type = 'string'
            elif dtype_type == 'float64' or dtype_type == 'int64':
                dtype_type = 'float'

            if dtype_column not in tag_list:
                dtype_dict['fieldKey'] = dtype_column
                dtype_dict['fieldType'] = dtype_type
                field_list.append(dtype_dict)

        if onlyFieldName:
            new_field_list = []
            for i in range(len(field_list)):
                new_field_list.append(field_list[i]['fieldKey'])

            field_list = new_field_list

        return field_list


    def create_database(self, bk_name):
       
        buckets_api = self.DBClient.buckets_api()

        if buckets_api.find_bucket_by_name(bucket_name=bk_name) == None:
            buckets_api.create_bucket(bucket_name=bk_name)


    def write_db_with_tags(self, df_data, bk_name, ms_name, tags_array, fields_array, batch_size=5000):

        with self.DBClient.write_api(write_options=WriteOptions(batch_size=batch_size)) as write_client:
            write_client.write(bucket=bk_name, record=df_data,
                               data_frame_measurement_name=ms_name, data_frame_tag_columns=tags_array)

    def ping(self):
        return self.DBClient.ping()

    def close_db(self) :
        self.DBClient.close()



    def write_db_large(self, bk_name, ms_name, data_frame):
        """
        Write large data to the influxdb
        """
        write_client = self.DBClient.write_api(write_options=ASYNCHRONOUS)
        if bk_name not in self.get_DBList():
            self.create_bucket(bk_name)
        
        df_count = len(data_frame.index)

        import math
        df_range = math.ceil(df_count/10000)

        for i in range(0, df_range):
            new_data_frame = data_frame.iloc[10000*i:10000*(i+1)-1]
            
            write_client.write(bucket=bk_name, record=new_data_frame, data_frame_measurement_name=ms_name)
            import time
            time.sleep(2)

        self.close_db()
        print("========== write success ==========")



    # def write_db_highCapacity(self, bk_name, ms_name, df) : 

    #     write_client = self.DBClient.write_api(write_options=ASYNCHRONOUS)    
        
    #     dataSize = len(df.index)

    #     if dataSize > 25000 : 
    #         chucksize = 25000
    #         import math
    #         share = math.ceil(dataSize / chucksize)
    #         x = 0
    #         y = chucksize                

    #         for idx in range(share) :
    #             chunkDF = df[x:y]
    #             write_client.write(bucket=bk_name, record=chunkDF, data_frame_measurement_name=ms_name)
    #             x = y
    #             y = x + chucksize
    #             print("dataSize :: ", dataSize, ", x ::", x, ", y ::" , y, ", share :: ", share)

    #     else : 
    #         write_client.write(bucket=bk_name, record=df, data_frame_measurement_name=ms_name)









## -------------------------------------- Influx Test --------------------------------------
# if __name__ == "__main__":
#     from KETIPreDataIngestion.KETI_setting import influx_setting_KETI as ins
#     test = influxClient(ins.CLUSTDataServer2)
#     db_name="air_indoor_아파트"
#     ms_name="ICW0W2000781"
    # db_name="air_indoor_경로당"
    # ms_name="ICL1L2000235"
    # start_time = '2021-05-01T00:00:00Z'
    # end_time = '2021-08-31T00:00:00Z'
    # db_name = "finance_korean_stock"
    # ms_name = "stock"
    # start_time = '2022-01-01T00:00:00Z'
    # end_time = '2022-02-28T00:00:00Z' 
    # number = 7
    # days = 7
    # tag_key = 'company'
    # tag_value = 'GS리테일'

    # aa = test.measurement_list(db_name)
    # print(aa)