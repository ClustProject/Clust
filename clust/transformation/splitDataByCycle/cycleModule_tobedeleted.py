import sys
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

sys.path.append("../")
sys.path.append("../../")

class CycleData():
    """
    # Description
        Cycle - Hour, Day, Week, Month, Year
    """
    def __init__(self):
        """
        # Description
            All data to be stored must start with '00:00:00'.
        """
        # self.start, self.end = self.getTimePointByDayUnit(data)
        # self.data = data[self.start: self.end] #(??)
        self.time_00 = datetime.strptime("00:00:00","%H:%M:%S").time()
        self.time_2300 = datetime.strptime("23:00:00","%H:%M:%S").time()
        self.time_2359 = datetime.strptime("23:59:59","%H:%M:%S").time()

    def getHourCycleSet(self, data, num, FullCycle):
        """
            # Description
                - Split the data by time ('00:00 to 59:59').
                - If num is 2 or more, split num*hour units.

            # Args
                - data (_pd.dataframe_) : timeseires data
                - num (_int_) : data cycle times
                - FullCycle (_bool_) : split complete or not
            
            # Returns
                - dataFrameCollectionResult (_List_) : data split by Hour Cycle

            # Note
            --------
            if set cycle Hour and num=3 --> hour*num --> split by 3Hours

        """
        hour_first = data.index[0]
        hour_last = data.index[-1]

        if hour_first.minute != 0 or hour_first.second != 0:
            hour_start = hour_first + timedelta(hours=1) - timedelta(minutes=hour_first.minute, seconds=hour_first.second)
        else:
            hour_start = hour_first

        hour_stop = hour_start + timedelta(hours=num) - timedelta(seconds=1)

        hour_one_stop = hour_start + timedelta(hours=1) - timedelta(seconds=1)
        hour_last_front = hour_last - timedelta(minutes=hour_last.minute, seconds=hour_last.second)

        hour_freq_count = len(data[hour_start:hour_one_stop])
        hour_last_count = len(data[hour_last_front:hour_last])

        if hour_freq_count != hour_last_count:
            hour_end = hour_last - timedelta(minutes=hour_last.minute, seconds=hour_last.second+1)
        else:
            hour_end = hour_last

        # hour_calcul = int((hour_end - hour_start).days*24 + ((hour_end - hour_start).seconds/3600) +1)
        # hour_count = math.ceil(hour_calcul / num)

        if hour_freq_count != 0:
            hour_count = int((len(data[hour_start:hour_end])/(hour_freq_count*num))) +1

            dataFrameCollectionResult = []
            for i in range(hour_count):
                dataframe_num_hour = data[hour_start:hour_stop]
                if len(dataframe_num_hour) != hour_freq_count*num:
                    if FullCycle == True and len(dataframe_num_hour) != 0:
                        dataFrameCollectionResult.append(dataframe_num_hour)
                    else:
                        pass
                else:
                    dataFrameCollectionResult.append(dataframe_num_hour)
                        
                hour_start = hour_stop + timedelta(seconds=1)
                hour_stop = hour_start + timedelta(hours=num) - timedelta(seconds=1)

                if hour_start + timedelta(hours=num) > hour_end:
                    hour_stop = hour_end
        else:
            dataFrameCollectionResult = None

        return dataFrameCollectionResult


    def getDayCycleSet(self, data, num, FullCycle):        
        """
            # Description
                - day 단위의 데이터 셋 리턴
                - Split the data by time ('00:00:00 ~ 23:59:59').
                - If num is 2 or more, split num*day units.

            # Args
                - data (_pd.dataframe_) : timeseires data
                - num (_int_) : data cycle times
                - FullCycle (_bool_) : split complete or not
            
            # Returns
                - dataFrameCollectionResult (_List_) : data split by Day Cycle

        """""
        # 첫 시간과 마지막 시간 구하기
        day_first = data.index[0]
        day_last = data.index[-1]

        # 들어온 dataframe의 첫 시간이 00:00:00이 아니면 다음날 00:00:00으로 변경
        if day_first.time() != self.time_00:
            day_start = day_first + timedelta(days=1) - timedelta(hours=day_first.hour, minutes=day_first.minute, seconds=day_first.second)
        else:
            day_start = day_first
        
        # start 시간에서 num만큼의 지난 날(범위를 지정할 때)
        day_stop = day_start + timedelta(days=num) - timedelta(seconds=1)

        # 마지막 주기의 데이터가 
        day_one_stop = day_start + timedelta(days=1) - timedelta(seconds=1)
        day_last_front = day_last - timedelta(hours=day_last.hour, minutes=day_last.minute, seconds=day_last.second)

        day_freq_count = len(data[day_start:day_one_stop])
        day_last_count = len(data[day_last_front:day_last])
        
        # 첫 주기의 데이터 개수와 마지막 주기의 데이터 개수가 다를 시 day_end 값 수정
        if day_freq_count != day_last_count:
            day_end = day_last - timedelta(hours=day_last.hour, minutes=day_last.minute, seconds=day_last.second+1)
        else:
            day_end = day_last
        if day_freq_count != 0:
            day_count = int((len(data[day_start:day_end])/(day_freq_count*num))) +1

            # 일 단위로 자른 데이터를 주기에 맞춰 dataframe에 저장 후, dataFrameCollectionResult에 append
            dataFrameCollectionResult = []
            for i in range(day_count):
                dataframe_num_day = data[day_start:day_stop]
                if len(dataframe_num_day) != day_freq_count*num:
                    if FullCycle == True and len(dataframe_num_day) != 0:
                        dataFrameCollectionResult.append(dataframe_num_day)
                    else:
                        pass
                else:
                    dataFrameCollectionResult.append(dataframe_num_day)
                        
                # 저장한 마지막 데이터 범위(23:59:59)에서 1초 추가하여 다음날(00:00:00)로 변경
                day_start = day_stop + timedelta(seconds=1)
                day_stop = day_start + timedelta(days=num) - timedelta(seconds=1)

                # 끝 데이터가 day_end보다 클 시, 원본 데이터의 마지막 날을 넘어가므로 day_end를 day_stop 값으로 설정
                if day_start + timedelta(days=num) > day_end:
                    day_stop = day_end
        else:
            dataFrameCollectionResult = None

        return dataFrameCollectionResult


    def getWeekCycleSet(self, data, num, FullCycle):     
        """
        # Description
            - Week 단위의 데이터 셋 리턴
            - Split the data by time ('Monday 00:00:00 ~ Sunday 23:59:59').
            - If num is 2 or more, split num*week units. 예를 들어서 num이 2일 경우, 2주 단위로 자름

        # Args
            - data (_pd.dataframe_) : timeseires data
            - num (_int_) : data cycle times
            - FullCycle (_bool_) : split complete or not
        
        # Returns
            - dataFrameCollectionResult (_List_) : data split by Day Cycle
            
        """""
        week_first = data.index[0]
        week_last = data.index[-1]
        
        # dataframe의 첫번째 데이터 처리
        if week_first.day_name() != 'Monday' and week_first.time() != self.time_00:
            week_start =  week_first + timedelta(weeks=1) - timedelta(days=week_first.dayofweek, hours=week_first.hour, minutes=week_first.minute, seconds=week_first.second)
        elif week_first.day_name() != 'Monday' and week_first.time() == self.time_00:
            week_start = week_first + timedelta(weeks=1) - timedelta(days=week_first.dayofweek)
        elif week_first.day_name() == 'Monday' and week_first.time() != self.time_00:
            week_start = week_first + timedelta(weeks=1) - timedelta(hours=week_first.hour, minutes=week_first.minute, seconds=week_first.second)
        else:
            week_start = week_first

        week_stop = week_start + timedelta(weeks=num) - timedelta(seconds=1)

        week_one_stop = week_start + timedelta(weeks=1) - timedelta(seconds=1)
        week_last_front = week_last - timedelta(days=week_last.dayofweek, hours=week_last.hour, minutes=week_last.minute, seconds=week_last.second)

        week_freq_count = len(data[week_start:week_one_stop])
        week_last_count = len(data[week_last_front:week_last])

        if week_freq_count != week_last_count:
            week_end = week_last - timedelta(days=week_last.dayofweek, hours=week_last.hour, minutes=week_last.minute, seconds=week_last.second+1)
        else:
            week_end = week_last

        if week_freq_count != 0:
            # week_count = math.ceil( ((week_end - week_start).days/7) / num)
            week_count = int((len(data[week_start:week_end])/(week_freq_count*num))) +1

            dataFrameCollectionResult = []
            for i in range(week_count):
                dataframe_num_week = data[week_start:week_stop]
                if len(dataframe_num_week) != week_freq_count*num:
                    if FullCycle == True and len(dataframe_num_week) != 0:
                        dataFrameCollectionResult.append(dataframe_num_week)
                    else:
                        pass
                else:
                    dataFrameCollectionResult.append(dataframe_num_week)
                        
                week_start = week_stop + timedelta(seconds=1)
                week_stop = week_start + timedelta(weeks=num) - timedelta(seconds=1)

                if week_start + timedelta(weeks=num) > week_end:
                    week_stop = week_end
        else:
            dataFrameCollectionResult = None

        return dataFrameCollectionResult


    def getMonthCycleSet(self, data, num, FullCycle):       
        """
        # Description
            - Month 단위의 데이터셋 리턴
            - Split the data by time ('1st 00:00:00 ~  last day 23:59:59').
            - If num is 2 or more, split num*month units.

        # Args
            - data (_pd.dataframe_): timeseires data
            - num (_int_): data cycle times
            - FullCycle (_bool_): split complete or not
        
        # Returns
            - dataFrameCollectionResult (_List_) : data split by Day Cycle

        """""
        month_first = data.index[0]
        month_last = data.index[-1]

        # 시작 월 설정
        if month_first.day != 1 and month_first.time() != self.time_00:
            month_start = month_first + relativedelta(months=1) - timedelta(days=month_first.day-1, hours=month_first.hour, minutes=month_first.minute, seconds=month_first.second)
        elif month_first.day != 1 and month_first.time() == self.time_00:
            month_start = month_first + relativedelta(months=1) - timedelta(days=month_first.day-1)
        elif month_first.day == 1 and month_first.time() != self.time_00:
            month_start = month_first + relativedelta(months=1) - timedelta(hours=month_first.hour, minutes=month_first.minute, seconds=month_first.second)
        else:
            month_start = month_first

        month_stop = month_start + relativedelta(months=num) - timedelta(seconds=1)
        # 마지막 데이터 프레임을 '23:59:59'로 맞춰준다
        month_last_end = month_last - relativedelta(days=month_last.day-1, hours=month_last.hour, minutes=month_last.minute, seconds=month_last.second+1)
        # 한 주기의 월 수
        month_freq_count = (month_stop.year - month_start.year)*12 + (month_stop.month - month_start.month) + 1
        # 현재 데이터 프레임의 월 수
        month_total_count =  (month_last_end.year - month_start.year)*12 + (month_last_end.month - month_start.month) +1
        # 총 월 수에 주기를 나눠 반복 횟수 지정
        month_div_num = (month_total_count // num)

        # 나머지가 있으면 남는 기간이 있다는 뜻
        if month_total_count % num == 0:
            month_last_front = month_start + relativedelta(months=num*(month_div_num-1))
            month_count = month_div_num
        else:
            month_last_front = month_start + relativedelta(months=num*month_div_num)
            month_count = month_div_num +1

        month_last_count = (month_last_end.year - month_last_front.year)*12 + (month_last_end.month - month_last_front.month) + 1
 
        # dataframe 마지막 데이터 설정
        if month_freq_count != month_last_count:
            month_end = month_last_end
        else:
            month_end = month_last
        if month_freq_count != 0:
            dataFrameCollectionResult = []
            for i in range(month_count):
                dataframe_num_month = data[month_start:month_stop]
                month_calcul = (month_stop.year - month_start.year)*12 + (month_stop.month - month_start.month) + 1

                if month_calcul != month_freq_count:
                    if FullCycle == True and len(dataframe_num_month) != 0:
                        dataFrameCollectionResult.append(dataframe_num_month)
                    else:
                        pass
                else:
                    dataFrameCollectionResult.append(dataframe_num_month)

                month_start = month_stop + timedelta(seconds=1)
                month_stop = month_start + relativedelta(months=num) - timedelta(seconds=1)

                if month_start + relativedelta(months=num) > month_end:
                    month_stop = month_end
        else:
            dataFrameCollectionResult = None

        return dataFrameCollectionResult



    def getYearCycleSet(self, data, num, FullCycle):
      
        """
        # Description
            - Year 단위의 데이터셋 리턴
            - Split the data by time ('01-01 00:00:00 ~  12-31 23:59:59').
            - If num is 2 or more, split num*year units.

        # Args
            - data (_pd.dataframe_): timeseires data
            - num (_int_): data cycle times
            - FullCycle (_bool_): split complete or not
        
        # Returns
            - dataFrameCollectionResult (_List_) : data split by Day Cycle
        """""
        year_first = data.index[0]
        year_last = data.index[-1]

        # 시작 데이터를 'xx-01-01 00:00:00' 으로 맞춰준다
        if year_first.strftime("%m-%d") != '01-01' and year_first.time()  != self.time_00:
            year_start = year_first + relativedelta(years=1) - relativedelta(months=year_first.month-1, days=year_first.day-1, hours=year_first.hour, minutes=year_first.minute, seconds=year_first.second)
        elif year_first.strftime("%m-%d") != '01-01' and year_first.time()  == self.time_00:
            year_start = year_first + relativedelta(years=1) - relativedelta(months=year_first.month-1, days=year_first.day-1)
        elif year_first.strftime("%m-%d") == '01-01' and year_first.time() != self.time_00:
            year_start = year_first + relativedelta(years=1) - timedelta(hours=year_first.hour, minutes=year_first.minute, seconds=year_first.second)
        else:
            year_start = year_first

        year_stop = year_start + relativedelta(years=num) - timedelta(seconds=1)
        # 마지막 데이터 프레임을 '12-31 23:59:59'로 맞춰준다
        # year_last_end = year_last - relativedelta(months=year_last.month-1 ,days=year_last.day-1, hours=year_last.hour, minutes=year_last.minute, seconds=year_last.second +1)
        if self.time_2300 <= year_last.time() <= self.time_2359 and year_last.strftime("%m-%d") == '12-31':
            year_last_end = year_last
        else:
            year_last_end = year_last - relativedelta(months=year_last.month-1 ,days=year_last.day-1, hours=year_last.hour, minutes=year_last.minute, seconds=year_last.second +1)   
        # 한 주기의 월 수
        year_freq_count = (year_stop.year - year_start.year) + 1
        # 현재 데이터 프레임의 월 수
        year_total_count =  (year_last_end.year - year_start.year) + 1
        # 총 월 수에 주기를 나눠 반복 횟수 지정
        year_div_num = (year_total_count // num)

        # 나머지가 있으면 남는 기간이 있다는 뜻
        if year_total_count % num == 0:
            year_last_front = year_start + relativedelta(years=num*(year_div_num-1))
            year_count = year_div_num
        else:
            year_last_front = year_start + relativedelta(years=num*year_div_num)
            year_count = year_div_num +1

        year_last_count = (year_last_end.year - year_last_front.year) + 1

        # dataframe 마지막 데이터 설정
        if year_freq_count != year_last_count:
            year_end = year_last_end
        else:
            year_end = year_last

        if year_freq_count != 0:
            dataFrameCollectionResult = []
            for i in range(year_count):
                dataframe_num_year = data[year_start:year_stop]
                year_calcul = (year_stop.year - year_start.year) + 1

                if year_calcul != year_freq_count:
                    if FullCycle == True and len(dataframe_num_year) != 0:
                        dataFrameCollectionResult.append(dataframe_num_year)
                    else:
                        pass
                else:
                    dataFrameCollectionResult.append(dataframe_num_year)

                year_start = year_stop + timedelta(seconds=1)
                year_stop = year_start + relativedelta(years=num) - timedelta(seconds=1)

                if year_start + relativedelta(years=num) > year_end:
                    year_stop = year_end

        else:
            dataFrameCollectionResult = None
            
        return dataFrameCollectionResult