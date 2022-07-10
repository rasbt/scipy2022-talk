import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from watermark import watermark

from helper_code.dataset import get_trip_advisor_datasetloaders
from helper_code.model import LightningRNN, PyTorchRNN


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

    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--num_devices", nargs="+", default="auto")

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

    print(watermark())
    print(watermark(packages="torch,pytorch_lightning,coral_pytorch"))

    parser = argparse.ArgumentParser()
    args = parse_cmdline_args(parser)

    torch.manual_seed(args.random_seed)

    # Architecture:
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256

    # Dataset specific:

    NUM_CLASSES = 5
    VOCAB_SIZE = 20000

    csv_path = os.path.join(args.data_path, "tripadvisor_balanced.csv")

    # Compute performance baselines
    train_loader, valid_loader, test_loader = get_trip_advisor_datasetloaders(
        csv_path=csv_path,
        random_seed=args.random_seed,
        vocab_size=VOCAB_SIZE,
        batch_size=args.batch_size,
        device="cpu",
    )

    all_test_labels = []
    for features, labels in test_loader:
        all_test_labels.append(labels)
    all_test_labels = torch.cat(all_test_labels)
    all_test_labels = all_test_labels.float()
    avg_prediction = torch.median(all_test_labels)  # median minimizes MAE
    baseline_mae = torch.mean(torch.abs(all_test_labels - avg_prediction))
    print(f"Baseline MAE: {baseline_mae:.2f}")

    # Initialize model
    pytorch_model = PyTorchRNN(
        input_dim=VOCAB_SIZE,  # len(TEXT.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    )

    lightning_model = LightningRNN(pytorch_model, learning_rate=args.learning_rate)

    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="valid_mae")
    ]  # save top 1 model
    logger = CSVLogger(
        save_dir=os.path.join(args.output_path, "logs/"), name="rnn-corn-tripadvisor"
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.num_devices,
        default_root_dir=args.output_path,
        strategy=args.strategy,
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
    print(f"Training took {runtime:.2f} min.")

    before = time.time()
    val_acc = trainer.test(dataloaders=valid_loader, ckpt_path="best")
    runtime = (time.time() - before) / 60
    print(f"Inference on the validation set took {runtime:.2f} min.")

    before = time.time()
    test_acc = trainer.test(dataloaders=test_loader, ckpt_path="best")
    runtime = (time.time() - before) / 60
    print(f"Inference on the test set took {runtime:.2f} min.")

    runtime = (time.time() - start_time) / 60
    print(f"The total runtime was {runtime:.2f} min.")

    print("Validation accuracy:", val_acc)
    print("Test accuracy:", test_acc)

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
    plt.savefig(os.path.join(args.output_path, "loss_plot.pdf"))

    df_metrics[["train_mae", "valid_mae"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="MAE"
    )
    plt.savefig(os.path.join(args.output_path, "mae_plot.pdf"))
