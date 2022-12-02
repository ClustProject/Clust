class TimeFeature():
    
    def __init__(self):
        pass
    
    def extendNewTimeFeatureByTimeindex(self, original_df):
        """
        This function generate new features(hour, day, month, day_of_week, week_of_year features), and extend original dataframe

        Example:
            >>> from clust.transformation.featureExtension.timeFeatureExtension import TimeFeature
            >>> TF = TimeFeature()
            >>> df_generated = TF.extendNewTimeFeatureByTimeindex(df)

        Args:
            original_df (DataFrame): original Input DataFrame with timeDataIndex

        Returns:
            DataFrame: extended_df - New dataFrmae with new time features(hour, day, month, day_of_week, week_of_year features)
        """ 
        if original_df.index.inferred_type == "datetime64":
            df_generated = (
                            original_df
                            .assign(hour = original_df.index.hour)
                            .assign(day = original_df.index.day)
                            .assign(month = original_df.index.month)
                            .assign(day_of_week = original_df.index.dayofweek)
                            .assign(week_of_year = original_df.index.week)
                        )
        else:
            print("Index typs is not datetime type")
        return df_generated

    import collections
    def extendWorkDayOffFeature(self, origin_df, holiday_list):
        """
        This function generate new feature "dayOff" and extend original dataframe.
        When feature value is 0, index time is dayOff (Holiday, Saturday, Sunday). when 1, index time is dayon.
        
        Example:
            >>> from clust.transformation.featureExtension.timeFeatureExtension import TimeFeature
            >>> TF = TimeFeature()
            >>> extended_df = TF.extendWorkDayOffFeature(df_features, us_holidays)

        Args:
            original_df (DataFrame): original Input DataFrame with timeDataIndex
            holiday_list (list): list of datetime

        Returns:
            DataFrame: extended_df - New dataFrmae with new time feature, "dayOff"
        """ 
        
        extended_df = origin_df.copy()

        if 'day_of_week' in extended_df:
            dayofweekFlag=True
            print("It has dayofweek column.")
        else:
            extended_df['day_of_week'] = extended_df.index.dayofweek

        extended_df['dayOff'] = 0
        extended_df.loc[extended_df['day_of_week'].isin([5, 6]), 'dayOff']=1
        extended_df.loc[extended_df.index.isin(holiday_list), 'dayOff']=1
   
        return extended_df

    def extendWorkTimeFeature(self, origin_df, workStartTime, workEndTime):
        """
        This function generate new feature "worktime" and extend original dataframe.
        When feature value is 0, index time is not working time.
        
        Example:
            >>> from clust.transformation.featureExtension.timeFeatureExtension import TimeFeature
            >>> TF = TimeFeature()
            >>> extended_df = TF.extendWorkTimeFeature(origin_df, workStartTime, workEndTime)

        Args:
            original_df (DataFrame): original Input DataFrame with timeDataIndex
            workStartTime (int): work start time
            workEndTime (int): work end time

        Returns:
            DataFrame: extended_df - New dataFrmae with new time feature, "worktime"
        """ 
        
        extended_df = origin_df.copy()
        #2. worktime_grade
        extended_df['worktime']= 1
        extended_df.loc[extended_df.index.hour < workStartTime, 'worktime']= 0
        extended_df.loc[extended_df.index.hour > workEndTime, 'worktime']= 0
        
        dayoffFlag=False
        if 'dayOff' in origin_df:
            extended_df.loc[extended_df['dayOff']== 0, 'worktime']= 0
        else:
            print("This DF does not have dayOff column.")
        return extended_df

def getHolidayList(country, startYear, endYear):
    if country == 'SouthKorea':
        from workalendar.asia import SouthKorea
        calendar = SouthKorea()
    elif country =='USA':
        from workalendar.usa import California
        calendar = California()
    else:
        from workalendar.europe import italy
        calendar = italy()


    holiday_list=[]
    for year in range(startYear, endYear+1):
        holiday_list = holiday_list + calendar.holidays(year)
    holiday_list = list(zip(*holiday_list))[0]
    return holiday_list