from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch





class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class AutoformerTimeSeriesPredictor(pl.LightningModule):
    def __init__(self, model_name='facebook/autoformer', sequence_length=128):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_length = sequence_length

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

dataset = TimeSeriesDataset(sequences, targets)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AutoformerTimeSeriesPredictor()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader)