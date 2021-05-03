from __future__ import print_function, division, absolute_import, unicode_literals

from abc import ABC, abstractmethod


############## Basis Class ##############

class RegularizerClass(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prox(self, *args, **kwargs):
        pass

    @abstractmethod
    def red(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval(self, *args, **kwargs):
        pass

    @abstractmethod
    def init(self, *args, **kwargs):
        pass
