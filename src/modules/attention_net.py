import torch


class AttentionNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size=128, similarity='inner_product'):
        super(AttentionNet, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = torch.nn.LSTM(
            input_size=dim_embeddings,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_rnn = torch.nn.LSTM(
            input_size=hidden_size*2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.metrixW = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear = torch.nn.Linear(hidden_size * 8, hidden_size * 2)

        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, context, context_lens, options, option_lens):

        batch_size = context.size(0)
        hidden_size = self.hidden_size

        context, (h_n, h_c) = self.rnn(context, None)  # (10, 23, 300) -> (10, 23, 128*2)
        context = self.dropout(context)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):    # options: (10, 5, 50, 300) -> (5, 10, 50, 300)

            option, (o_n, o_c) = self.rnn(option, None)         # (10, 50, 128*2)
            option = self.dropout(option)

            alphas = context.bmm(option.transpose(1, 2))            # (10, 23, 128*2)*(10, 128*2, 50) = (10, 23, 50)
            alphas = self.softmax(alphas)                           # (10, 23, 50)
            alphas = alphas.transpose(1, 2)                         # (10, 50, 23)
            new_context = alphas.bmm(context)                       # (10, 50, 128*2)
            new_context = self.dropout(new_context)

            interaction = torch.cat((option, new_context, option*new_context, option-new_context), 2)
            # (10, 50, 128*2*4)
            interaction = self.linear(interaction.view(-1, hidden_size*8))     # (10*50, 128*2*4)
            interaction = interaction.view(batch_size, -1, hidden_size*2)        # (10, 50, 128*2)

            attn_option, _ = self.attn_rnn(interaction)     # (10, 50, 128*2)
            option_last = attn_option.max(1)[0]             # (10, 128*2)

            attn_context, _ = self.attn_rnn(context)        # (10, 23, 128*2)
            context_last = attn_context.max(1)[0]           # (10, 128*2)

            c_out = self.metrixW(context_last).unsqueeze(1)   # c_out(10, 128*2) -> (10, 1, 128*2)
            o_out = option_last.unsqueeze(2)  # (10, 128*2, 1)
            logit = c_out.bmm(o_out).squeeze(2).squeeze(1)   # (10, 1, 1) -> (10)
            logits.append(logit)       # -> (5, 10)
        logits = torch.stack(logits, 1)  # (10, 5)

        return logits
