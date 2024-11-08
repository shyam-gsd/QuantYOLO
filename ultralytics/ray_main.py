import sys
import os

import pyarrow
import ray.train

from ray.tune import Callback

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

os.environ["TUNE_ORIG_WORKING_DIR"] = "/home/shyam/PycharmProjects/QuantYOLO/ultralytics/"
sys.path.append('/home/shyam/PycharmProjects/QuantYOLO/')
sys.path.append('/home/shyam/PycharmProjects/QuantYOLO/ultralytics/ultralytics')
sys.path.append('/home/shyam/PycharmProjects/QuantYOLO/brevitas/src')
from pathlib import Path

import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ultralytics import YOLO
from ray.tune import stopper

epoch_cnt = 0


def on__train_epoch_end(trainer):
    print(trainer.start_epoch)


def update_best_model_path():
    best_model_path = Path("/home/shyam/PycharmProjects/QuantYOLO/ultralytics/runs/detect/train92/weights/best.pt")
    dir_runs = Path("/home/shyam/PycharmProjects/QuantYOLO/ultralytics/runs/detect")

    fols = sorted([f for f in dir_runs.iterdir() if f.is_dir() and f.name.startswith("train")])

    best_map = 0.0
    for f in fols:
        try:
            csv = pd.read_csv(f / "results.csv", sep=",")
            map = csv["    metrics/mAP50-95(B)"].max()
            if map > best_map:
                if Path(f / "weights/best.pt").exists():
                    best_map = map
                    est_model_path = f / "weights/best.pt"
        except:
            pass
    print("updated path to " + str(best_model_path))
    return best_model_path


# Define training function for YOLO
def train_yolo(config):
    model_path = update_best_model_path()
    model = YOLO(model_path,task="detect")  # Or any other YOLO model config
    model.add_callback("on_epoch_end", on__train_epoch_end)
    model.train(
        data="coco128.yaml",
        epochs=100,
        lr0=config["lr0"],
        momentum=config["momentum"],
        batch=config["batch"],
        val=True,
    )




algo = HyperOptSearch(metric="metrics/mAP50(B)", mode="max",points_to_evaluate=[{"lr0":0.01578,"momentum":0.93099,"batch":64},{"lr0":0.011,"momentum":0.95638,"batch":64}])
algo = ConcurrencyLimiter(algo, max_concurrent=4)

# Define hyperparameter search space
search_space = {
    "lr0": tune.loguniform(1e-5, 1e-2),
    "momentum": tune.uniform(0.8, 0.95),
    "batch": tune.choice([16, 32, 64, 128])  # Batch size
}

# Configure the ASHA scheduler for efficient pruning
asha_scheduler = ASHAScheduler(
    metric="metrics/mAP50(B)",
    mode="max",
    grace_period=10,
    reduction_factor=5,
    time_attr="epoch",
    max_t=500
)

stopper = stopper.TrialPlateauStopper(metric="metrics/mAP50(B)", mode="max", grace_period=7,std=0.02,num_results=5)

exp_dir = "./experiments"
exp_run = 0
for dir in os.listdir(exp_dir):
    name, exp = dir.split("_")
    exp_run += 1

exp_path = exp_dir + "/exp_" + str(exp_run) + "/"
os.makedirs(exp_path)

trainable_with_resources = tune.with_resources(train_yolo,{"cpu": 8, "gpu": 1})


# Run the hyperparameter tuning with Ray Tune
tuner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=1,
        scheduler=asha_scheduler,
        search_alg=algo,
        reuse_actors=True
    ),
    run_config=ray.train.RunConfig(
        stop=stopper,
        storage_path=Path(exp_path).resolve().as_posix(),
        storage_filesystem=pyarrow.fs.LocalFileSystem(use_mmap=False),
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_at_end=False,
        ),
        failure_config=ray.train.FailureConfig(
            fail_fast=True,
        ),

    ),
)

res = tuner.fit()
