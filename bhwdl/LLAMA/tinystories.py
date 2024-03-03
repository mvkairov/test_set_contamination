import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    def __init__(self, data_file, sp_model_prefix=None, max_length=256):
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model', )
        print("tokenizer ok")
        with open(data_file, encoding="utf-8") as file:
            texts = list(map(lambda x: x.strip(), file.readlines()))
        self.texts = texts
        print("texts ok")
        self.indices = self.sp_model.encode(self.texts)
        print("encoding ok")

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts):
        return self.sp_model.encode(texts)

    def ids2text(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().tolist()
        return self.sp_model.decode(ids)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        indices = [self.bos_id] + self.indices[item][:self.max_length - 2] + [self.eos_id]
        length = len(indices)
        pad = torch.full((self.max_length,), self.pad_id, dtype=torch.int64)
        pad[:length] = torch.tensor(indices)
        return torch.tensor(pad), torch.tensor(length)
