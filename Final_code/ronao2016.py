"""Functions for training Convolutional Neural Network (CNN)"""
from typing import Any, Dict
from models.base import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPool2D
from tensorflow.keras import regularizers


class Ronao2016(BaseModel):
    """
    @article{ronao2016human,
        title={Human activity recognition with smartphone sensors using deep learning neural networks},
        author={Ronao, Charissa Ann and Cho, Sung-Bae},
        journal={Expert systems with applications},
        volume={59},
        pages={235--244},
        year={2016},
        publisher={Elsevier}
    }
    """

    def __init__(self, config: Dict, LOG_DIR: str, logger: Any):
        super(Ronao2016, self).__init__(config=config, LOG_DIR=LOG_DIR, logger=logger)

    def build_model(self, input_shape, n_classes):
        drp_out_dns = self.parameters["drp_out_dns"]
        nb_dense = self.parameters["nb_dense"]
        kernel_regularizer = regularizers.l2(0.00005)

        self.model = Sequential()
        self.model.add(Input(input_shape))
        self.model.add(
            Conv2D(
                96,
                kernel_size=(9, 1),
                kernel_regularizer=kernel_regularizer,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )
        )
        self.model.add(MaxPool2D(pool_size=(3, 1)))
        self.model.add(
            Conv2D(
                192,
                kernel_size=(9, 1),
                kernel_regularizer=kernel_regularizer,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )
        )
        self.model.add(MaxPool2D(pool_size=(3, 1)))
        self.model.add(
            Conv2D(
                192,
                kernel_size=(9, 1),
                kernel_regularizer=kernel_regularizer,
                strides=(1, 1),
                padding="same",
                activation="relu",
            )
        )
        self.model.add(MaxPool2D(pool_size=(3, 1)))
        self.model.add(Flatten())
        if False:  # Adaptive:
            pass
            # self.model.add(
            #     DimensionAdaptivePoolingForSensors(
            #         pool_list, operation="max", name="DAP", forRNN=False
            #     )
            # )
        else:
            self.model.add(
                Dense(
                    nb_dense,
                    kernel_regularizer=kernel_regularizer,
                    activation="relu",
                    name="act_dns",
                )
            )
        self.model.add(Dropout(drp_out_dns, name="act_drp_out"))
        self.model.add(Dense(n_classes, activation="softmax", name="act_smx"))
