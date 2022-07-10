import pytorch_lightning as pl
import torch
import torchmetrics
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss


# Regular PyTorch Module
class PyTorchMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes, loss_mode):
        super().__init__()

        # num_classes is used by the corn loss function
        self.num_classes = num_classes

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit

        # -------------------------------------------------------------
        if loss_mode == "corn":
            output_layer = torch.nn.Linear(hidden_units[-1], num_classes - 1)
        elif loss_mode == "crossentropy":
            output_layer = torch.nn.Linear(hidden_units[-1], num_classes)
        else:
            raise ValueError("loss_mode must be 'corn' or 'crossentropy'.")
        # -------------------------------------------------------------

        self.loss_mode = loss_mode

        all_layers.append(output_layer)
        self.model = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x


# LightningModule that receives a PyTorch model as input
class LightningMLP(pl.LightningModule):
    def __init__(self, model, learning_rate, loss_mode):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_mode = loss_mode
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        # -------------------------------------------------------------
        if self.loss_mode == "corn":
            loss = corn_loss(logits, true_labels, num_classes=self.model.num_classes)
            predicted_labels = corn_label_from_logits(logits)
        elif self.loss_mode == "crossentropy":
            loss = torch.nn.functional.cross_entropy(logits, true_labels)
            predicted_labels = torch.argmax(logits, dim=1)
        else:
            raise ValueError("loss_mode must be 'corn' or 'crossentropy'.")
        # -------------------------------------------------------------

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_mae(predicted_labels, true_labels)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_mae(predicted_labels, true_labels)
        self.log(
            "valid_mae", self.valid_mae, on_epoch=True, on_step=False, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_mae(predicted_labels, true_labels)
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
