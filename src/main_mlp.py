import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from watermark import watermark

from helper_code.dataset import get_cement_dataloaders
from helper_code.model import LightningMLP, PyTorchMLP


def parse_cmdline_args(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default="auto")

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--data_path", type=str, default="./data")

    parser.add_argument("--learning_rate", type=float, default=0.0005)

    parser.add_argument(
        "--mixed_precision", type=str, choices=("true", "false"), default="false"
    )

    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=3)

    parser.add_argument("--num_devices", nargs="+", default="auto")

    parser.add_argument(
        "--loss_mode", type=str, choices=("corn", "crossentropy"), default="corn"
    )

    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--device_numbers", type=str, default="")

    parser.add_argument("--random_seed", type=int, default=-1)

    parser.add_argument("--strategy", type=str, default="")

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    if not args.strategy:
        args.strategy = None

    if args.num_devices != "auto":
        args.num_devices = int(args.num_devices[0])
    if args.device_numbers:
        args.num_devices = [int(i) for i in args.device_numbers.split(",")]

    d = {"true": True, "false": False}

    args.mixed_precision = d[args.mixed_precision]
    if args.mixed_precision:
        args.mixed_precision = 16
    else:
        args.mixed_precision = 32

    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parse_cmdline_args(parser)

    log_out = os.path.join(args.output_path, f"{args.loss_mode}_traininglog.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_out), logging.StreamHandler()],
    )

    logging.info(watermark())
    logging.info(watermark(packages="torch,pytorch_lightning,coral_pytorch"))

    torch.manual_seed(args.random_seed)

    csv_path = os.path.join(args.data_path, "cement_strength.csv")

    # Compute performance baselines

    train_loader, valid_loader, test_loader = get_cement_dataloaders(
        csv_path=csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_test_labels = []
    for features, labels in test_loader:
        all_test_labels.append(labels)
    all_test_labels = torch.cat(all_test_labels)
    all_test_labels = all_test_labels.float()
    avg_prediction = torch.median(all_test_labels)  # median minimizes MAE
    baseline_mae = torch.mean(torch.abs(all_test_labels - avg_prediction))
    logging.info(f"Baseline MAE: {baseline_mae:.2f}")

    # Initialize model
    pytorch_model = PyTorchMLP(
        input_size=features.shape[1],
        hidden_units=(40, 20),
        num_classes=np.bincount(all_test_labels).shape[0],
        loss_mode=args.loss_mode,
    )

    lightning_model = LightningMLP(
        pytorch_model, learning_rate=args.learning_rate, loss_mode=args.loss_mode
    )

    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="valid_mae")
    ]  # save top 1 model
    logger = CSVLogger(
        save_dir=os.path.join(args.output_path, "lightning_logs/"),
        name=f"{args.loss_mode}-mlp-cement",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.num_devices,
        default_root_dir=args.output_path,
        strategy=args.strategy,
        logger=logger,
        precision=args.mixed_precision,
        deterministic=False,
        log_every_n_steps=10,
    )

    start_time = time.time()
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    train_time = time.time()
    runtime = (train_time - start_time) / 60
    logging.info(f"Training took {runtime:.2f} min.")

    before = time.time()
    val_acc = trainer.test(dataloaders=valid_loader, ckpt_path="best")
    runtime = (time.time() - before) / 60
    logging.info(f"Inference on the validation set took {runtime:.2f} min.")

    before = time.time()
    test_acc = trainer.test(dataloaders=test_loader, ckpt_path="best")
    runtime = (time.time() - before) / 60
    logging.info(f"Inference on the test set took {runtime:.2f} min.")

    runtime = (time.time() - start_time) / 60
    logging.info(f"The total runtime was {runtime:.2f} min.")

    logging.info("Validation accuracy:", val_acc)
    logging.info("Test accuracy:", test_acc)

    # Make plots

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)

    df_metrics[["train_loss", "valid_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    plt.savefig(os.path.join(args.output_path, f"{args.loss_mode}_loss_plot.pdf"))

    df_metrics[["train_mae", "valid_mae"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="MAE"
    )
    plt.savefig(os.path.join(args.output_path, f"{args.loss_mode}_mae_plot.pdf"))
