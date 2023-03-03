import abc
from typing import Any


class BaseRegressionModel(abc.ABC):
    """
    
    """
    @abc.abstractmethod
    def train(self) -> None:
        pass

    @abc.abstractmethod
    def test(self) -> Any:
        pass

    @abc.abstractmethod
    def inference(self) -> Any:
        pass

    # @abc.abstractmethod
    # def get_test_result(self) -> Any:
    #     pass

    # @abc.abstractmethod
    # def get_inference_result(self) -> Any:
    #     pass

    @abc.abstractmethod
    def export_model(self) -> Any:
        pass

    @abc.abstractmethod
    def save_model(self) -> None:
        pass

    @abc.abstractmethod
    def load_model(self) -> None:
        pass

    @abc.abstractmethod
    def create_trainloader(self) -> Any:
        pass

    @abc.abstractmethod
    def create_testloader(self) -> Any:
        pass

    @abc.abstractmethod
    def create_inferenceloader(self) -> Any:
        pass
