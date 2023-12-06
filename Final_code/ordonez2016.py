"""Functions for training Convolutional Neural Network (CNN)"""
from typing import Any, Dict
from models.base import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv2D, Input, Reshape


class Ordonez2016(BaseModel):
    """
    @article{ordonez2016deep,
        title={Deep convolutional and {LSTM} recurrent neural networks for multimodal wearable activity recognition},
        author={Ord{\'o}{\~n}ez, Francisco and Roggen, Daniel},
        journal={Sensors},
        volume={16},
        number={1},
        pages={115},
        year={2016},
        publisher={Multidisciplinary Digital Publishing Institute}
    }
    """

    def __init__(self, config: Dict, LOG_DIR: str, logger: Any):
        super(Ordonez2016, self).__init__(config=config, LOG_DIR=LOG_DIR, logger=logger)

    def build_model(self, input_shape, n_classes):
        nb_filters = self.parameters["nb_filters"]
        drp_out_dns = self.parameters["drp_out_dns"]
        nb_dense = self.parameters["nb_dense"]

        inp = Input(input_shape)

        x = Conv2D(
            nb_filters, kernel_size=(5, 1), strides=(1, 1), padding="valid", activation="relu"
        )(inp)
        x = Conv2D(
            nb_filters, kernel_size=(5, 1), strides=(1, 1), padding="valid", activation="relu"
        )(x)
        x = Conv2D(
            nb_filters, kernel_size=(5, 1), strides=(1, 1), padding="valid", activation="relu"
        )(x)
        x = Conv2D(
            nb_filters, kernel_size=(5, 1), strides=(1, 1), padding="valid", activation="relu"
        )(x)
        x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
        act = LSTM(nb_dense, return_sequences=True, activation="tanh", name="lstm_1")(x)
        act = Dropout(drp_out_dns, name="dot_1")(act)
        act = LSTM(nb_dense, activation="tanh", name="lstm_2")(act)
        act = Dropout(drp_out_dns, name="dot_2")(act)
        out_act = Dense(n_classes, activation="softmax", name="act_smx")(act)

        self.model = Model(inputs=inp, outputs=out_act)
