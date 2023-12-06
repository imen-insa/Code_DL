"""Functions for training Convolutional Neural Network (CNN)"""
from typing import Any, Dict
from models.base import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Input, MaxPool2D, Flatten


class Ignatov2018(BaseModel):
    """
    @article{ignatov2018real,
        title={Real-time human activity recognition from accelerometer data using Convolutional Neural Networks},
        author={Ignatov, Andrey},
        journal={Applied Soft Computing},
        volume={62},
        pages={915--922},
        year={2018},
        publisher={Elsevier}
    }
    """

    def __init__(self, config: Dict, LOG_DIR: str, logger: Any):
        super(Ignatov2018, self).__init__(config=config, LOG_DIR=LOG_DIR, logger=logger)

    def build_model(self, input_shape, n_classes):
        inp = Input(input_shape)
        x = Conv2D(196, kernel_size=(16, 6), strides=(1, 6), padding="same", activation="relu")(inp)
        x = MaxPool2D(pool_size=(4, 1), padding="same")(x)
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.95, name="act_drp_out")(x)
        out_act = Dense(n_classes, activation="softmax", name="act_smx")(x)

        self.model = Model(inputs=inp, outputs=out_act)
