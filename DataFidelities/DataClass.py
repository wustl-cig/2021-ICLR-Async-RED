from abc import ABC, abstractmethod

class DataClass(ABC):
	
    @abstractmethod
    def grad(self,x):
        pass