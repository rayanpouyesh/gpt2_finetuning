import os
import logging
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from asteroid.models import DPRNNTasNet
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

# تنظیمات اولیه logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

torch.set_float32_matmul_precision("medium")


# Custom Dataset class for 2 speakers with segmentation and padding
class TwoSpeakerDataset(Dataset):
    def __init__(self, mixture_dir, sources_dir, sample_rate=8000, segment_length=4.0):
        self.mixture_dir = mixture_dir
        self.sources_dir = sources_dir
        self.sample_rate = sample_rate
        self.segment_length = int(segment_length * sample_rate)  # 4 seconds
        self.mixture_files = [f for f in os.listdir(mixture_dir) if f.endswith('.wav')]
        self.all_segments = self._precompute_segments()

    def _precompute_segments(self):
        all_segments = []
        for idx, file in enumerate(self.mixture_files):
            mixture_file = os.path.join(self.mixture_dir, file)
            mixture, _ = torchaudio.load(mixture_file)
            num_segments = max(1, -(-mixture.shape[1] // self.segment_length))  # Ceiling division
            for seg_idx in range(num_segments):
                start = seg_idx * self.segment_length
                end = start + self.segment_length
                all_segments.append((idx, start, end))
        return all_segments

    def __len__(self):
        return len(self.all_segments)

    def __getitem__(self, seg_idx):
        idx, start, end = self.all_segments[seg_idx]
        mixture_file = os.path.join(self.mixture_dir, self.mixture_files[idx])
        mixture, _ = torchaudio.load(mixture_file)
        mixture = mixture.squeeze(0)  # Remove channel dimension

        if mixture.shape[0] < end:
            mixture = F.pad(mixture, (0, end - mixture.shape[0]), value=0.0)
        else:
            mixture = mixture[start:end]

        sources = []
        for i in range(1, 3):
            source_file = os.path.join(self.sources_dir, f"s{i}", f"{idx + 1}.wav")
            source, _ = torchaudio.load(source_file)
            source = source.squeeze(0)

            if source.shape[0] < end:
                source = F.pad(source, (0, end - source.shape[0]), value=0.0)
            else:
                source = source[start:end]

            sources.append(source)

        sources = torch.stack(sources, dim=0)  # Shape: (2, time)
        return mixture, sources


# Create the dataset and data loader
mixture_dir = "./New folder (6)/mix"
sources_dir = "./New folder (6)"

all_mixture_files = [f for f in os.listdir(mixture_dir) if f.endswith('.wav')]
train_files, val_files = train_test_split(all_mixture_files, test_size=0.2, random_state=42)

# Create separate dataset instances
train_dataset = TwoSpeakerDataset(mixture_dir, sources_dir)
val_dataset = TwoSpeakerDataset(mixture_dir, sources_dir)

# Update dataset instances to use split data
train_dataset.mixture_files = train_files
val_dataset.mixture_files = val_files

if __name__ == "__main__":
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    # Initialize the model
    model = DPRNNTasNet(n_src=2, n_blocks=8, n_repeats=6, masknet_kwargs={"dropout": 0.2})

    # Define the loss function
    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define the system using Asteroid
    from asteroid.engine import System

    system = System(model, optimizer, loss, train_loader, val_loader)

    # Set up ModelCheckpoint to save the model during training
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # Initialize the Trainer
    trainer = Trainer(
        max_epochs=500,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback],
        num_nodes=1,
        deterministic=True,
        gradient_clip_val=1.0
    )

    # Start the training
    trainer.fit(system)
