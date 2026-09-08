from src.baselines import Baselines


class FixedPrefixProbeRunner(Baselines):
    def load_data(self) -> None:
        return None

    def preprocess(self, input_text: str) -> str:
        return input_text

    def postprocess(self, input_text: str, output_text: str) -> str:
        return output_text

    def eval(self) -> None:
        return None
