import torch
from torch import nn

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index =\
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range =\
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(config.hidden_size,
                            config.hidden_size,
                            num_layers=2,
                            bias=True,
                            batch_first=True,
                            dropout=config.hidden_dropout_prob,
                            bidirectional=True)

    def forward(self, hidden_output, attention_mask):
        sequences_lengths = torch.sum(attention_mask, dim=-1)
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(hidden_output, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)
        lstm_outputs, _ = self.lstm(packed_batch, None)
        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs,
                                                           batch_first=True)
        reordered_outputs = lstm_outputs.index_select(0, restoration_idx)
        batch_max_lengths = sorted_lengths[0]
        return reordered_outputs, batch_max_lengths