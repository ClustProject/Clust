import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

def get_timestep_feature(data, timestep_criteria = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}):
    """
    # Description
        A function that adds a "TimeStep" column constructed according to the input timeStep.            
        - Since the function is classified based on Hour, the Input data time frequency must be Hour, Minute, or Second.
        - Used when the period of data time information is less than 1 hour.

    Args:
        data (_pd.dataframe_) : Time series data
        timestep_criteria (_Dictionary_) : TimeStep criteria information
        

    >>> timestep_criteria = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}
    
    Returns:
        _pd.dataframe_ : data( Time sereis data with "TimeStep" column)

    """
    timestep = timestep_criteria["step"]
    timelabel = timestep_criteria["label"]
    
    data["TimeStep"] = np.array(None)

    for n in range(len(timestep)-1):
        data.loc[data[(data.index.hour >= timestep[n])&(data.index.hour < timestep[n+1])].index, "TimeStep"] = timelabel[n]

    return data

def split_data_by_timestep(data, timestep = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}):
        """
        # Description
            Split the data by TimeStep.

        Args:
            data (_pd.dataframe_) : Time series data
            timestep_criteria (_Dictionary_) : TimeStep criteria infromation
            

        >>> timestep_criteria = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}

        Returns:
            _Dictionary_ : split_data_by_timestep(Return value composed of dataframes divided according to each label of timestep.)
        """
        
        # Get data with timestep feature
        data = get_timestep_feature(data, timestep)
        
        # Split Data by Timestep
        split_data_by_timestep = {}
        
        for label in timestep["label"]:
            split_data_by_timestep[label] = data[label == data["TimeStep"]].drop(["TimeStep"], axis=1)
        
        return split_data_by_timestep
    

   