import pytorch_lightning as pl
import torch
import torchmetrics


# Regular PyTorch Module
class PyTorchRNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_classes):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)

        # CORN output layer ------------------------------------------
        # Regular classifier would use num_classes instead of
        # num_classes-1 below
        self.output_layer = torch.nn.Linear(hidden_dim, num_classes - 1)
        # ------------------------------------------------------------

        self.num_classes = num_classes

    def forward(self, text, text_length):
        # text dim: [sentence len, batch size]

        embedded = self.embedding(text)
        # embedded dim: [sentence len, batch size, embed dim]

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_length.to("cpu")
        )

        packed_output, (hidden, cell) = self.rnn(packed)
        # output dim: [sentence len, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        output = self.output_layer(hidden)
        logits = output.view(-1, (self.num_classes - 1))

        return logits


# LightningModule that receives a PyTorch model as input
class LightningRNN(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.input_dim = model.input_dim
        self.embedding_dim = model.embedding_dim
        self.hidden_dim = model.hidden_dim
        self.num_classes = model.num_classes

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    # (Re)Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, text, text_length):
        return self.model(text, text_length)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):

        # These next 3 steps are unique and look a bit tricky due to
        # how Torchtext's BucketIterator prepares the batches
        # and how we use an LSTM with packed & padded text
        # Also, .TEXT_COLUMN_NAME and .LABEL_COLUMN_NAME
        # depend on the CSV file columns of the data file we load later.
        features, text_length = batch.TEXT_COLUMN_NAME
        true_labels = batch.LABEL_COLUMN_NAME

        logits = self(features, text_length)

        # Use CORN loss ---------------------------------------------------
        # A regular classifier uses:
        # loss = torch.nn.functional.cross_entropy(logits, true_labels)
        loss = corn_loss(logits, true_labels, num_classes=self.model.num_classes)
        # -----------------------------------------------------------------

        # CORN logits to labels -------------------------------------------
        # A regular classifier uses:
        # predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = corn_label_from_logits(logits)
        # -----------------------------------------------------------------

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss, batch_size=true_labels.shape[0])
        self.train_mae(predicted_labels, true_labels)
        self.log(
            "train_mae",
            self.train_mae,
            on_epoch=True,
            on_step=False,
            batch_size=true_labels.shape[0],
        )
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss, batch_size=true_labels.shape[0])
        self.valid_mae(predicted_labels, true_labels)
        self.log(
            "valid_mae",
            self.valid_mae,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=true_labels.shape[0],
        )

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_mae(predicted_labels, true_labels)
        self.log(
            "test_mae",
            self.test_mae,
            on_epoch=True,
            on_step=False,
            batch_size=true_labels.shape[0],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
