class RefineFrequency():
    """ Refine Data with the static frequency
    """
    def __init__(self):
        pass

    def get_RefinedData(self, data, freq=None):
        """ This function makes new data with the static description frequency according to the freq parameter status. 
        
        Args:
            data (DataFrame): input data
            freq ([None| DateOffset|Timedelta|str]): Frequency of output data
            
        Note
        --------
        If None, this module infers the data frequency and redefines it.


        Returns:
            DataFrame: NewDataframe output with static description frequency without redunency
        
        Example:

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

        Args:
            data (DataFrame): input data

        Returns:
            DataFrame, DateOffset: NewDataframe output, inferred_frequency
        
        Example:

            >>> output, new_frequency = RefineFrequency().get_RefinedDatawithInferredFreq(data)
        """
        
        inffered_freq = self.get_frequencyWith3DataPoints(data)
        self.output = self.make_static_frequencyData(data, inffered_freq)
        return self.output, inffered_freq
    
    def get_RefinedDatawithStaticFreq(self, data, freq):
        """ This function generates data with the static inference frequency.

        Args:
            data (DataFrame): input data
            freq (DateOffset): frequency of data to be newly 

        Returns:
            DataFrame: NewDataframe output

        Example:

            >>> output = RefineFrequency().get_RefinedDatawithStaticFreq(data, '30S')
        """
        
        self.output = self.make_static_frequencyData(data, freq)
        return self.output
    
    def get_RefinedDataSetwithStaticFreq(self, dataSet, freq=None):
        """ This function makes new dataSet with the static description frequency according to the freq parameter status. 
        
        Args:
            data (Dictionary): input data
            freq (DateOffset, Timedelta or str): Frequency of output data
            
        Note
        -----------
        If None, this module infers the data frequency and redefines it.


        Returns:
            Dictionary: NewDataframeSet output with static description frequency without redunency 
        
        Example:

            >>> output = RefineFrequency().get_RefinedDataSetwithStaticFreq(dataSet, None)
        """
        newDataSet={}
        for index in dataSet:
            newDataSet[index] = RefineFrequency().get_RefinedDatawithStaticFreq(dataSet[index], freq)

        return newDataSet


    def make_static_frequencyData(self, data, freq):
        """ This function makes data with static frequency. if freq is None, just pass the data

        Args:
            data (DataFrame): input data
            freq (DateOffset, Timedelta or str): frequency of data to be newly generated
            
        Returns:
            Dictionary: NewDataframe output

        Example:

            >>> output = RefineFrequency().make_static_frequencyData(data, '30S')
        """
        data_static_frequency = data.copy()
        data_static_frequency = data_static_frequency.sort_index()
        if freq:
            data_static_frequency = data_static_frequency.resample(freq).mean()
            data_static_frequency = data_static_frequency.asfreq(freq=freq)
            
        return data_static_frequency
    
    def get_frequencyWith3DataPoints(self, data):
        """ this function inferrs description frequency of input data
        
        Args:
            data (DataFrame): input data
            
        Returns:
            DateOffset: estimated_freq

        Example:

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
        
        if not estimated_freq:
            try:
                estimated_freq = (data.index[1]-data.index[0])
            except :
                print("예외가 발생했습니다. data : ", data)
    
        return estimated_freq