import torch


class BestNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size=128, similarity='inner_product'):
        super(BestNet, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = torch.nn.LSTM(
            input_size=dim_embeddings,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_rnn = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.metrixW = torch.nn.Linear(hidden_size*2, hidden_size*2)
        self.linear = torch.nn.Linear(hidden_size*8, hidden_size)

        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, context, context_lens, options, option_lens):

        batch_size = context.size(0)
        hidden_size = self.hidden_size

        context, (h_n, h_c) = self.rnn(context, None)  # (32, 50, 300) -> (32, 50, 128*2)
        context = self.dropout(context)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):    # options: (10, 5, 50, 300) -> (5, 10, 50, 300)

            option, (o_n, o_c) = self.rnn(option, None)         # (32, 300, 128*2)
            option = self.dropout(option)

            alphas = context.bmm(option.transpose(1, 2))            # (50,256)x(256,300) = (b, 50, 300)
            alphas = self.softmax(alphas)                           # (b, 50, 300)
            alphas = alphas.transpose(1, 2)                         # (b, 300, 50)
            new_context = alphas.bmm(context)                       # (b, 300, 128*2)
            new_context = self.dropout(new_context)

            alphas2 = option.bmm(context.transpose(1, 2))           # (300,256)x(256,50) = (b, 300, 50)
            alphas2 = self.softmax(alphas2)                         # (b, 300, 50)
            alphas2 = alphas2.transpose(1, 2)                       # (b, 50, 300)
            new_option = alphas2.bmm(option)                        # (b, 50, 128*2)
            new_option = self.dropout(new_option)

            option_interact = torch.cat((option, new_context, option * new_context, option-new_context), 2)
            # (b, 300, 128*2*4)
            option_interact = self.linear(option_interact.view(-1, hidden_size*8))   # (b*300, 128)
            option_interact = option_interact.view(batch_size, -1, hidden_size)    # (b, 300, 128)

            context_interact = torch.cat((context, new_option, context*new_option, context-new_option), 2)
            # (b, 50, 128*2*4)
            context_interact = self.linear(context_interact.view(-1, hidden_size*8))  # (b*50, 128)
            context_interact = context_interact.view(batch_size, -1, hidden_size)   # (b, 50, 128)

            attn_option, _ = self.attn_rnn(option_interact)  # (b, 300, 128) -> (b, 300, 128*2)
            option_last = attn_option.max(1)[0]  # (b, 128*2)

            attn_context, _ = self.attn_rnn(context_interact)
            context_last = attn_context.max(1)[0]  # (b, 128*2)

            c_out = self.metrixW(context_last).unsqueeze(1)   # c_out(b, 128*2) -> (b, 1, 128*2)
            o_out = option_last.unsqueeze(2)  # (10, 128*2, 1)
            logit = c_out.bmm(o_out).squeeze(2).squeeze(1)   # (10, 1, 1) -> (10)

            logits.append(logit)       # -> (5, 10)
        logits = torch.stack(logits, 1)  # (10, 5)

        return logits

