import torch


class MetrixNet(torch.nn.Module):

    def __init__(self, n_feature=256, n_hidden=256):
        super(MetrixNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)

    def forward(self, context, option):
        context_out = self.hidden(context)
        logit = context_out.mm(option)
        return logit

