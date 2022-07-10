import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    def __init__(self, feature_array, label_array, dtype=np.float32):
        self.features = feature_array.astype(dtype)
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.features.shape[0]


def get_cement_dataloaders(csv_path, batch_size, num_workers):
    data_df = pd.read_csv(csv_path)
    data_df["response"] = data_df["response"] - 1  # labels should start at 0
    data_labels = data_df["response"]
    data_features = data_df.loc[:, ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]]

    # Split into
    # 70% train, 10% validation, 20% testing

    X_temp, X_test, y_temp, y_test = train_test_split(
        data_features.values,
        data_labels.values,
        test_size=0.2,
        random_state=1,
        stratify=data_labels.values,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=1, stratify=y_temp
    )

    # Standardize features
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_valid_std = sc.transform(X_valid)
    X_test_std = sc.transform(X_test)

    train = ArrayDataset(X_train_std, y_train)
    valid = ArrayDataset(X_valid_std, y_valid)
    test = ArrayDataset(X_test_std, y_test)

    train_dataloader = DataLoader(
        train, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )

    val_dataloader = DataLoader(valid, batch_size=batch_size, num_workers=num_workers)

    test_dataloader = DataLoader(test, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader
