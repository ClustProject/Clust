class RefineFrequency():
    """ Refine Data with the static frequency
    """
    def __init__(self):
        pass

    def get_RefinedData(self, data, freq=None):
        """ This function makes new data with the static description frequency according to the freq parameter status. 
        
        :param data: input data
        :type data: DataFrame 
        :param freq: Frequency of output data. If None, this module infers the data frequency and redefines it.
        :type freq: [None| DateOffset|Timedelta|str]
        
        :return: NewDataframe output with static description frequency without redunency 
        :rtype: DataFrame
        
        example
            >>> output = RefineFrequency().get_RefinedData(data, None)
        """
        self.data = data
        self.freq = freq
        if not self.freq:
            self.output, self.freq = self.get_RefinedDatawithInferredFreq(data)
        else:
            self.output = self.get_RefinedDatawithStaticFreq(data, self.freq)
        return self.output

    def get_RefinedDatawithInferredFreq(self, data):
        """ This function generates data with inferred static inference frequency.

        :param data: input data
        :type data: DataFrame 

        :return: NewDataframe output, inferred_frequency
        :rtype: DataFrame, DateOffset
        
        example
            >>> output, new_frequency = RefineFrequency().get_RefinedDatawithInferredFreq(data)
        """
        
        inffered_freq = self.get_frequencyWith3DataPoints(data)
        self.output = self.make_staticFrequencyData(data, inffered_freq)
        return self.output, inffered_freq
    
    def get_RefinedDatawithStaticFreq(self, data, freq):
        """ This function generates data with the static inference frequency.

        :param data: input data
        :type data: DataFrame 
        :param freq: frequency of data to be newly 
        :type freq: DateOffset 
        
        :return: NewDataframe output
        :rtype: DataFrame

        example
            >>> output = RefineFrequency().get_RefinedDatawithStaticFreq(data, '30S')
        """
        
        self.output = self.make_staticFrequencyData(data, freq)
        return self.output
    
    def get_RefinedDataSetwithStaticFreq(self, dataSet, freq=None):
        """ This function makes new dataSet with the static description frequency according to the freq parameter status. 
        
        :param data: input data
        :type data: DataFrameSet (dictionary)
        :param freq: Frequency of output data. If None, this module infers the data frequency and redefines it.
        :type freq: DateOffset, Timedelta or str
        
        :return: NewDataframeSet output with static description frequency without redunency 
        :rtype: DataFrameSet (dictionary)
        
        example
            >>> output = RefineFrequency().get_RefinedDataSetwithStaticFreq(dataSet, None)
        """
        newDataSet={}
        for index in dataSet:
            newDataSet[index] = RefineFrequency().get_RefinedDatawithStaticFreq(dataSet[index], freq)

        return newDataSet


    def make_staticFrequencyData(self, data, freq):
        """ This function makes data with static frequency.

        :param data: input data
        :type data: DataFrame
        :param freq: frequency of data to be newly generated
        :type freq: DateOffset, Timedelta or str

        :return: NewDataframe output
        :rtype: DataFrame

        example
            >>> output = RefineFrequency().make_staticFrequencyData(data, '30S')
        """
        data_staticFrequency = data.copy()
        data_staticFrequency = data_staticFrequency.sort_index()
        data_staticFrequency = data_staticFrequency.resample(freq).mean()
        data_staticFrequency = data_staticFrequency.asfreq(freq=freq)
        return data_staticFrequency
    
    def get_frequencyWith3DataPoints(self, data):
        """ this function inferrs description frequency of input data

        :param data: input data
        :type data: DataFrame

        :return: estimated_freq
        :rtype: DateOffset

        example
            >>> estimated_freq  = RefineFrequency().get_frequencyWith3DataPoints(data)
        
        """
        if len(data)> 3:
            # Simply compare 2 intervals from 3 data points.
            # And get estimated frequency.
            
            inferred_freq1 = (data.index[1]-data.index[0])
            inferred_freq2 = (data.index[2]-data.index[1])
           
            if inferred_freq1 == inferred_freq2:
                estimated_freq = inferred_freq1
            else:
                inferred_freq1 = (data.index[-1]-data.index[-2])
                inferred_freq2 = (data.index[-2]-data.index[-3])
                if inferred_freq1 == inferred_freq2:
                    estimated_freq = inferred_freq1
                else :
                    estimated_freq = None
        else:
            estimated_freq = None
        
        # TODO Modify it 
        # if etstmated_freq is None, it infers using only two data points.
        
        ##TODO:  책임님께 질문 2022.09.29 소.이.
        if not estimated_freq:
            try:
                estimated_freq = (data.index[1]-data.index[0])
            except :
                print("예외가 발생했습니다. data : ", data)
    
        return estimated_freq