import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import AutoModel
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassConfusionMatrix


class CommaModel(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        bert_dim = 264
        self.hidden_size = bert_dim
        self.num_class = num_class
        self.pretrained_transformer = AutoModel.from_pretrained('DeepPavlov/distilrubert-tiny-cased-conversational-v1')
        self.linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=num_class
        )

        # for param in self.pretrained_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor) -> torch.Tensor:
        # add dummy batch for single sample
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        # x = [batch_size, seq_len, emb_dim]
        x = self.pretrained_transformer(x, attention_mask=attn_masks)[0]
        x = self.linear(x)
        return x


class ModelLightning(pl.LightningModule):
    def __init__(
            self,
            model,
            criterion,
            t_max,
            lr: float = 0.05,
            weight_decay: float = 0.001,
        ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.t_max = t_max
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.save_hyperparameters(ignore=['model', 'criterion'])

        device = next(self.model.parameters()).device
        self.f1_metric = F1Score(task='multiclass', num_classes=model.num_class).to(device)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=model.num_class).to(device)

    def forward(self, x, attn_mask):
        return self.model(x, attn_mask)
    
    def evaluate(self, batch, mode: str) -> tuple:
        x = batch['x']
        y = batch['y']
        attn_mask = batch['attn_mask']

        output = self.model(x, attn_mask)
        
        # eval loss
        loss = self.criterion(
            output.view(-1, output.shape[2]),
            y.view(-1),
        )

        if mode == 'valid':
            metrics = self.f1_metric(F.softmax(output.view(-1, output.shape[2]), dim=-1), y.view(-1))
            return loss, metrics
        elif mode == 'test':
            self.conf_matrix.update(output.argmax(dim=-1).view(-1), y.view(-1))
        return loss, None
    
    def training_step(self, batch, batch_idx):
        train_loss, _ = self.evaluate(batch, 'train')
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        valid_loss, valid_metric = self.evaluate(batch, 'valid')
        self.log("valid_loss", valid_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_metric", valid_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return valid_loss
    
    def test_step(self, batch, batch_idx):
        test_loss, _ = self.evaluate(batch, 'test')
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.t_max,
        )
        
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            ]
        )