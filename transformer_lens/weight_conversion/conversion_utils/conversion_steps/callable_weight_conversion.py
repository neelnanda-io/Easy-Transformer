from .base_weight_conversion import BaseWeightConversion


class CallableWeightConversion(BaseWeightConversion):
    def __init__(self, convert_callable: callable):
        super().__init__()
        self.convert_callable = convert_callable

    def handle_conversion(self, input_value: dict):
        return self.convert_callable(input_value)
