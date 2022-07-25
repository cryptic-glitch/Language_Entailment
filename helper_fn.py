import numpy as np
import torch
import dataclasses


#################################################
############### Metric ##########################
@dataclasses.dataclass
class Metrics:
    accuracy: float = 0.
    correct_preds: int = 0
    total_labels: int = 0

    def compute_accuracy(self, preds: torch.Tensor, gt: torch.tensor):
        self.total_labels += len(gt)
        for ind, val in enumerate(preds):
            if torch.argmax(preds[ind]) == torch.argmax(gt[ind]):
                self.correct_preds += 1

    def reset_metrics(self):
        self.correct_preds = 0
        self.total_labels = 0
        self.accuracy = 0.

    def get_acc(self):
        return self.correct_preds / self.total_labels


###################################################
############ Collate ##############################
class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids[0]) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                torch.tensor(np.hstack(s.tolist() + (batch_max - len(s[0])) * [self.tokenizer.pad_token_id])) for s in
                output["input_ids"]]
            output["attention_mask"] = [torch.tensor(np.hstack(s.tolist() + (batch_max - len(s[0])) * [0])) for s in
                                        output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.stack(output["input_ids"])
        output["attention_mask"] = torch.stack(output["attention_mask"])
        targets = torch.tensor(output.pop("target"), dtype=torch.float)
        return output, targets
