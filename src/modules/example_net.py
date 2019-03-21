import torch
import pdb


class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size=64, similarity='inner_product'):
        super(ExampleNet, self).__init__()

        self.rnn = torch.nn.LSTM(
            input_size=dim_embeddings,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout=0.2,
        )

        self.attn_rnn = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            # dropout=0.2,
            batch_first=True,
        )

        self.metrixW = torch.nn.Linear(hidden_size, hidden_size)
        self.attnMetrix = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.attn_dropout = torch.nn.Dropout(0.1)

    def forward(self, context, context_lens, options, option_lens):
        context, (h_n, h_c) = self.rnn(context, None)  # (10, 23, 300) -> (10, 23, 128)
        # context_last = context[:, -1, :]               # (10, 128)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):    # options: (10, 5, 50, 300) -> (5, 10, 50, 300)

            option, (o_n, o_c) = self.rnn(option, None)     # (10, 50, 128)
            # option_last = option_out[:, -1, :]              # (10, 128)
            drop_context = self.attn_dropout(context)
            temp = self.attnMetrix(drop_context)              # (10, 23, 128)
            alphas = temp.bmm(option.transpose(1, 2))    # (10, 23, 128).bmm(10, 128, 50) = (10, 23, 50)
            alphas = self.softmax(alphas)                           # (10, 23, 50)
            alphas = alphas.transpose(1, 2)                         # (10, 50, 23)
            new_context = alphas.bmm(context)                   # (10, 50, 128)
            interaction = torch.cat((option, new_context, option*new_context, option-new_context), 1)
            # (10, 50*4, 128)

            attn_option, _ = self.attn_rnn(interaction)  # (10, 50*4, 128) -> (10, 50*4, 128)
            option_last = attn_option[:, -1, :]  # (10, 128)

            attn_context, _ = self.attn_rnn(context)
            context_last = attn_context[:, -1, :]  # (10, 128)

            c_out = self.metrixW(context_last).unsqueeze(1)   # c_out(10,128) -> (10, 1, 128)
            o_out = option_last.unsqueeze(2)  # (10, 128, 1)
            logit = c_out.bmm(o_out).squeeze(2).squeeze(1)   # (10, 1, 1) -> (10)
            logits.append(logit)       # -> (5, 10)
        logits = torch.stack(logits, 1)  # (10, 5)

        return logits

