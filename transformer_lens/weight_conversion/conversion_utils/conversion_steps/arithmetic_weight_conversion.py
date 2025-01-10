from enum import Enum

import torch

from .base_weight_conversion import BaseWeightConversion


class OperationTypes(Enum):
    ADDITION = 0
    SUBTRACTION = 1
    MULTIPLICATION = 2
    DIVISION = 3


class ArithmeticWeightConversion(BaseWeightConversion):
    def __init__(self, operation: OperationTypes, value: float | int | torch.Tensor, input_filter: callable|None = None):
        super().__init__(input_filter=input_filter)
        self.operation = operation
        self.value = value

    def handle_conversion(self, input_value):
        match self.operation:
            case OperationTypes.ADDITION:
                return input_value + self.value
            case OperationTypes.SUBTRACTION:
                return input_value - self.value
            case OperationTypes.MULTIPLICATION:
                return input_value * self.value
            case OperationTypes.DIVISION:
                return input_value / self.value
