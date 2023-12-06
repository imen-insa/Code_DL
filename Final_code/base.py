"""Functions for training Convolutional Neural Network (CNN)"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from src.utils import plot_learning_history, plot_model
from src.keras_callback import create_callback
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.utils import plot_confusion_matrix
from src.utils import round


class BaseModel:
    """
    Base class for models
    """

    def __init__(self, config: Dict, LOG_DIR: str, logger: Any):
        self.parameters = config["model"]
        self.dataset_name = config["dataset"]
        self.LOG_DIR = LOG_DIR
        self.logger = logger
        self.model = None

    def train_and_predict(self, ds_train, ds_valid, ds_test) -> Dict[str, Dict[str, List[Any]]]:
        """Train CNN
        Args:
            X_train, X_valid, X_test: input signals of shape
                (num_samples, window_size, num_channels, 1)
            y_train, y_valid, y_test: onehot-encoded labels
        Returns:
            pred_train: train prediction
            pred_valid: train prediction
            pred_test: train prediction
            model: trained best model
        """

        input_shape, output_shape = [(x.shape, y.shape) for x, y in ds_train.take(1)][0]

        ds_train = ds_train.batch(self.parameters["batch_size"])
        ds_valid = ds_valid.batch(self.parameters["batch_size"])
        ds_test = ds_test.batch(self.parameters["batch_size"])

        self.build_model(input_shape=input_shape, n_classes=output_shape[0])
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.parameters["lr"]),
            metrics=["accuracy"],
        )

        with open(f"{self.LOG_DIR}/model_summary.txt", "w") as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + "\n"))
        plot_model(self.model, path=f"{self.LOG_DIR}/model.png")

        callbacks = create_callback(
            log_dir=self.LOG_DIR,
            verbose=1,
            epochs=self.parameters["epochs"],
            mcl_name=f"{self.dataset_name}-{self.__class__.__name__}",
            mcl_params=self.parameters,
        )

        fit = self.model.fit(
            ds_train,
            epochs=self.parameters["epochs"],
            verbose=self.parameters["verbose"],
            validation_data=ds_valid,
            callbacks=callbacks,
        )

        plot_learning_history(fit=fit, path=f"{self.LOG_DIR}/history.png")
        best_model = keras.models.load_model(f"{self.LOG_DIR}/trained_model.h5")

        pred_train = self.model.predict(ds_train)
        pred_valid = self.model.predict(ds_valid)
        pred_test = self.model.predict(ds_test)

        scores: Dict[str, Dict[str, List[Any]]] = {
            "logloss": {"train": [], "valid": [], "test": []},
            "accuracy": {"train": [], "valid": [], "test": []},
            "precision": {"train": [], "valid": [], "test": []},
            "recall": {"train": [], "valid": [], "test": []},
            "f1": {"train": [], "valid": [], "test": []},
            "cm": {"train": [], "valid": [], "test": []},
            "per_class_f1": {"train": [], "valid": [], "test": []},
        }
        for pred, ds, mode in zip(
            [pred_train, pred_valid, pred_test],
            [ds_train, ds_valid, ds_test],
            ["train", "valid", "test"],
        ):
            loss, acc = best_model.evaluate(ds, verbose=0)
            pred = pred.argmax(axis=1)
            y = np.array([np.array(v) for v in ds.unbatch().map(lambda _, yy: yy)])
            y = y.argmax(axis=1)
            scores["logloss"][mode].append(loss)
            scores["accuracy"][mode].append(acc)
            scores["precision"][mode].append(precision_score(y, pred, average="macro"))
            scores["recall"][mode].append(recall_score(y, pred, average="macro"))
            scores["f1"][mode].append(f1_score(y, pred, average="macro"))
            scores["cm"][mode].append(confusion_matrix(y, pred, normalize="true"))
            scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))

        np.save(f"{self.LOG_DIR}/scores.npy", scores)
        df_scores = pd.DataFrame([])
        for mode in ["train", "valid", "test"]:
            res_dict = {}
            for metric in ["logloss", "accuracy", "precision", "recall", "f1"]:
                res_dict[metric] = [round(np.mean(scores[metric][mode]))]
            df = pd.DataFrame(res_dict, index=[mode])
            df_scores = pd.concat([df_scores, df])
        print("--------------------------------------------------------")
        print(df_scores)
        print("--------------------------------------------------------")
        df_scores.to_csv(f"{self.LOG_DIR}/scores.csv")
        np.save(f"{self.LOG_DIR}/test_oof.npy", np.mean(pred_test, axis=0))

        # Plot confusion matrix
        plot_confusion_matrix(
            cms=scores["cm"],
            labels=[
                "LAYING",
                "WALKING",
                "WALKING_UPSTAIRS",
                "WALKING_DOWNSTAIRS",
                "SITTING",
                "STANDING",
            ],
            path=f"{self.LOG_DIR}/confusion_matrix.png",
        )

        return scores
