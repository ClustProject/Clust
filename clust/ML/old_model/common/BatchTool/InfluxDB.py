import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.tool import model

class BatchTraining():
    """
    BatchTraining for Influx DB

    """
    def __init__(self, dbClient):
        """
        :param dbClient: instance of influxdb client
        :type dbClient: instance of influxdb client
        """
        self.DBClient = dbClient

    def setTrainer(self, trainer):
        """
        set setTrainer

        **Example**::
            >>> from Clust.clust.ML.brits.train import BrisTrainer
            >>> Brits = BritsTrainer()
            >>> modelTrainParameter = None
            >>> Brits.setTrainParameter(modelTrainParameter)
            >>> trainer.setTrainMethod(Brits)

        :param trainer: trainer instance
        :type trainer: class instance
        """
        self.trainer = trainer

    def setBatchParameter(self, dataIngestionParameter, trainMethod):
        """
        set setBatchParameter

        """
        self.dataIngestionParameter = dataIngestionParameter
        self.trainMethod = trainMethod

    def batchTrain(self):
        """
        train model by batch style. It can make model for all DB or only one MS.

        **Example**::
            >>> trainer.setParameter(dataIngestionParameter, modelInfo)
            >>> trainer.batchTrain()   

        """
        db_name = self.dataIngestionParameter['db_name']
        #MSColumn
        if "ms_name" in self.dataIngestionParameter:
            self.trainerForMSColumn()
        #DBMSColumn
        else:
            msList = self.DBClient.measurement_list(db_name)
            for ms_name in msList:
                self.dataIngestionParameter['ms_name'] = ms_name
                self.trainerForMSColumn()

    def trainerForMSColumn(self):
        """
        train model for only one MS by each column  

        """
        ms_name = self.dataIngestionParameter['ms_name']
        db_name = self.dataIngestionParameter['db_name']

        if 'duration' in self.dataIngestionParameter:
            duration = self.dataIngestionParameter['duration']
            start_time = duration['start_time']
            end_time = duration['end_time']
            df = self.DBClient.get_data_by_time(start_time, end_time, db_name, ms_name)
        
        elif 'number' in self.dataIngestionParameter:
            number = self.dataIngestionParameter['number'] 
            df = self.DBClient.get_data_front_by_num(number, db_name, ms_name)
        
        else:
            number = 2000
            df = self.DBClient.get_data_front_by_num(number, db_name, ms_name)
                
        for column_name in df.columns: 
            trainDataPathList = [db_name, ms_name, column_name]#, str(bind_params)]
            modelFilePath = model.get_model_path(trainDataPathList, self.trainMethod)

            self.trainer.trainModel(df[[column_name]],  modelFilePath)
    
    