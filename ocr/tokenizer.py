import torch

from ctcdecode import CTCBeamDecoder


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


def get_char_map(alphabet):
    """Make from string alphabet character2int dict.
    Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list, merge_repeated=True):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary tokens."""
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip blank symbols, oov tokens and repeated characters
                if (
                    char_enc != self.char_map[OOV_TOKEN]
                    and char_enc != self.char_map[CTC_BLANK]
                    # (idx > 0) condition to avoid selecting [-1] item
                    and not (merge_repeated and idx > 0
                             and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class BeamSearcDecoder:
    def __init__(self, alphabet, lm_path):
        self.tokenizer = Tokenizer(alphabet)
        char_map = self.tokenizer.char_map
        labels = [
            k for k, v in sorted(char_map.items(), key=lambda item: item[1])
        ]
        self.decoder = CTCBeamDecoder(
               labels=labels,
               model_path=lm_path,
               alpha=0.6,
               beta=1.1,
               cutoff_top_n=10,
               cutoff_prob=1,
               beam_width=10,
               num_processes=6,
               blank_id=0,
               log_probs_input=True)

    def __call__(self, output):
        beam_results, _, _, out_lens = \
            self.decoder.decode(output.permute(1, 0, 2))
        encoded_texts = []
        for beam_result, out_len in zip(beam_results, out_lens):
            encoded_texts.append(
                beam_result[0][:out_len[0]].numpy()
            )
        text_preds = self.tokenizer.decode(encoded_texts, merge_repeated=False)
        return text_preds


class BestPathDecoder:
    def __init__(self, alphabet):
        self.tokenizer = Tokenizer(alphabet)

    def __call__(self, output):
        pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
        text_preds = self.tokenizer.decode(pred)
        return text_preds
