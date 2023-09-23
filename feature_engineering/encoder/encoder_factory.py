from feature_engineering.encoder.encoders import (
    OneHot,
    Ordinal,
    Frequency,
    Mean,
    OneHotFrequent,
    WoE,
)

ENCODERS = {
    "one_hot_encoder": OneHot,
    "ordinal_encoder": Ordinal,
    "frequency_encoder": Frequency,
    "mean_encoder": Mean,
    "woe_encoder": WoE,
    "ohe_frequent_encoder": OneHotFrequent,
}


class EncoderFactory:
    @staticmethod
    def get_encoder(encoder_type):
        if encoder_type not in ENCODERS:
            raise ValueError(
                f"Invalid Encoder type: {encoder_type}. "
                f"Supported encoders: {list(ENCODERS.keys())}"
            )

        return ENCODERS[encoder_type]()
