import abc
from typing import Any


class BaseRegressionModel(abc.ABC):
    """
    Base model class to build a ML model for the regression task.
    CLUST 플랫폼의 regression task 용 모델 클래스(e.g., RNNClust, CNN1DClust, ...) 생성시 상속받아야 할 base class 입니다.
    Base class에 추상 메서드로 데코레이트된 메서드는 반드시 정의되어야 합니다.
    """
    @abc.abstractmethod
    def train(self) -> None:
        """
        학습을 위한 ``train`` 메서드 입니다.
        criterion(loss 함수), optimizer 정의를 포함하여 자유로운 방식으로 self.model을 학습할 수 있으나, 
        train/validation data loader는 create_trainloader 메서드에서 정의하고, 학습을 위한 파라미터와 함께 입력인자로 전달받도록 합니다.
        학습 과정의 마지막에는 self.model 에 최종 weights를 로드하도록 합니다.

        Args:
            params (dict): parameters for train
            train_loader (Dataloader): train data loader
            valid_loader (Dataloader): validation data loader
            num_epochs (integer): the number of train epochs
            device (string): device for train
        """
        pass

    @abc.abstractmethod
    def test(self) -> Any:
        """
        테스트를 위한 ``test`` 메서드 입니다.
        load_model 메서드 정의에 따라 weights가 로드된 self.model을 테스트하며,
        test data loader는 create_testloader 메서드에서 정의하고, 테스트를 위한 파라미터와 함께 입력인자로 전달받도록 합니다.
        테스트 후에는 결과 데이터와 metric을 return 하도록 합니다.
         
        Args:
            params (dict): parameters for test
            test_loader (DataLoader): data loader
            device (string): device for test

        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
            mse (float): mean square error
            mae (float): mean absolute error 
        """
        pass

    @abc.abstractmethod
    def inference(self) -> Any:
        """
        추론을 위한 ``inference`` 메서드 입니다.
        load_model 메서드 정의에 따라 weights가 로드된 self.model로 추론을 수행하며,
        inference data loader는 create_inferenceloader 메서드에서 정의하고, 추론을 위한 파라미터와 함께 입력인자로 전달받도록 합니다.
        추론 후에는 결과 데이터를 return 하도록 합니다.


        Args:
            params (dict): parameters for inference
            inference_loader (DataLoader): inference data loader
            device (string): device for inference

        Returns:
            preds (ndarray): Inference result data
        """
        pass

    # @abc.abstractmethod
    # def get_test_result(self) -> Any:
    #     pass

    # @abc.abstractmethod
    # def get_inference_result(self) -> Any:
    #     pass

    @abc.abstractmethod
    def export_model(self) -> Any:
        """
        학습 결과 모델을 반환하기 위한 ``export_model`` 메서드 입니다.
        현재 self.model 객체를 return 하도록 합니다.


        Returns:
            self.model (Object): current model object
        """
        pass

    @abc.abstractmethod
    def save_model(self) -> None:
        """
        학습 결과 모델을 저장하기 위한 ``save_model`` 메서드 입니다.
        save_path 를 입력 인자로 받아 해당 경로에 특정 형식으로 저장하며, 동일 형식을 고려한 load_model 정의가 필요합니다.


        Args:
            save_path (string): path to save model
        """
        pass

    @abc.abstractmethod
    def load_model(self) -> None:
        """
        학습된 모델을 로드하기 위한 ``load_model`` 메서드 입니다.
        model_file_path 를 입력 인자로 받아 해당 경로에 저장된 모델을 로드하며, save_model 에서 정의한 형식과 호환되도록 정의해야 합니다.


        Args:
            model_file_path (string): path to load saved model
        """
        pass

    @abc.abstractmethod
    def create_trainloader(self) -> Any:
        """
        학습을 위한 train/validation data loader 정의를 위한 ``create_trainloader`` 메서드 입니다.
        학습에 필요한 data loader 정의 인자들을 입력으로 받고, train 메서드에서 받아 바로 학습에 활용할 수 있도록 정의합니다.
        DataLoader 형식의 trian_loader, valid_loader 를 return 하도록 합니다.


        Args:
            batch_size (integer): batch size
            train_x (dataframe): train X data
            train_y (dataframe): train y data
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data
            window_num (integer): slice window number

        Returns:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): validation data loader
        """
        pass

    @abc.abstractmethod
    def create_testloader(self) -> Any:
        """
        테스트를 위한 test data loader 정의를 위한 ``create_testloader`` 메서드 입니다.
        테스트에 필요한 data loader 정의 인자들을 입력으로 받고, test 메서드에서 받아 바로 테스트에 활용할 수 있도록 정의합니다.
        DataLoader 형식의 test_loader 를 return 하도록 합니다.


        Args:
            batch_size (integer): batch size
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data

        Returns:
            test_loader (DataLoader): test data loader
        """
        pass

    @abc.abstractmethod
    def create_inferenceloader(self) -> Any:
        """
        추론을 위한 inference data loader 정의를 위한 ``create_inferenceloader`` 메서드 입니다.
        추론에 필요한 data loader 정의 인자들을 입력으로 받고, inference 메서드에서 받아 바로 테스트에 활용할 수 있도록 정의합니다.
        DataLoader 형식의 inference_loader 를 return 하도록 합니다.


        Args:
            batch_size (integer): 
            x_data (dataframe): inference X data
            window_num (integer): slice window number
        
        Returns:
            inference_loader (DataLoader) : inference data loader
        """
        pass
