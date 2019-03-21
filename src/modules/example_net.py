import torch
import pdb


class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size=128, bidirectional=False, similarity='inner_product'):
        super(ExampleNet, self).__init__()

        self.rnn = torch.nn.LSTM(
            input_size=dim_embeddings,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout=0.2,
            bidirectional=bidirectional
        )
        self.metrixW = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                                       hidden_size * 2 if bidirectional else hidden_size)

    def forward(self, context, context_lens, options, option_lens):
        context_out, (h_n, h_c) = self.rnn(context, None)  # (10, 23, 300) -> (10, 23, 128)  h_n/c: (1, 10, 128)
        context_out_last = context_out[:, -1, :]               # (10, 128)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):    # options: (10, 5, 50, 300) -> (5, 10, 50, 300)
                                                                # ^ from the first option in every batch
            option_out, (o_n, o_c) = self.rnn(option, None)     # (10, 50, 128)
            option_out_last = option_out[:, -1, :]                   # (10, 128)

            # output (10, 50, 128) view (10, 50*128)  (10, 128)
            # input (10, 23, 128) view (10, 23*128)   (10, 23, 128)
            # 每一個option字對23個context_out字跑linear，把得到23維(attention)的vector過softmax後和原本的context相乘
            # for i, o in output: i(23, 128) o(50, 128)  de-batch
            #     for output in o:
            #

            # context_out_last(10, 128),   option_out_last(10, 128)
            c_out = self.metrixW(context_out_last).unsqueeze(1)   # c_out(10,128) -> (10, 1, 128)
            o_out = option_out_last.unsqueeze(2)  # (10, 128, 1)
            logit = c_out.bmm(o_out).squeeze(2).squeeze(1)   # (10, 1, 1) -> (10)
            logits.append(logit)       # -> (5, 10)
        logits = torch.stack(logits, 1)  # (10, 5)

        return logits

