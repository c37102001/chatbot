import torch


class RnnNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size=128, similarity='inner_product'):
        super(RnnNet, self).__init__()

        self.rnn = torch.nn.LSTM(
            input_size=dim_embeddings,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.metrixW = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, context, context_lens, options, option_lens):

        context, (h_n, h_c) = self.rnn(context, None)  # (32, 50, 300) -> (32, 50, 128)
        context_last = context[:, -1, :]               # (32, 128)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):    # options: (32, 5, 50, 300) -> (5, 32, 50, 300)

            option, (o_n, o_c) = self.rnn(option, None)     # (32, 300, 128)
            option_last = option[:, -1, :]              # (32, 128)

            c_out = self.metrixW(context_last).unsqueeze(1)   # c_out(32, 128) -> (32, 1, 128)
            o_out = option_last.unsqueeze(2)  # (32, 128, 1)
            logit = c_out.bmm(o_out).squeeze(2).squeeze(1)   # (32, 1, 1) -> (32)
            logits.append(logit)       # -> (5, 32)
        logits = torch.stack(logits, 1)  # (32, 5)

        return logits
