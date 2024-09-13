import torch
import transformers
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.save_hyperparameters()

        # 문자열로 표현된 loss와 optimizer를 함수로 변환
        self.model_name = CFG['model']['model_name']
        self.lr = float(CFG['train']['learning_rate'])
        self.loss_func = eval(CFG['train']['LossF'])()
        # self.optim은 configure_optimizers에서 사용
        self.optim = eval(CFG['train']['optim'])


        ## CFG의 model_name으로 설정된 모델 불러오기
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)


    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # 기존코드
        # self.log("train_loss", loss)

        # 에포크 단위로 로그 기록
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # 기존코드
        # self.log("val_loss", loss)
        # self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        # 에포크 단위로 로그 기록
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()), on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 기존코드
        # self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        # 에포크 단위로 로그 기록
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()), on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)
        return optimizer