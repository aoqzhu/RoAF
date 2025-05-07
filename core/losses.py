from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sigma = args['base']['sigma']
        self.moe = args['base']['moe']
        self.MSE_Fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()

    def forward(self, out, label):
        l_sp = self.loss_fn(out['sentiment_preds'], label['sentiment_labels'])
        moe_loss = out["moe_loss"]
        loss = + self.sigma * l_sp + self.moe * moe_loss

        return {'loss': loss, 'l_sp': l_sp, 'moe_loss': moe_loss}
