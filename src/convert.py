import torch
from torch import nn
import argparse
from esm.pretrained import load_model_and_alphabet_local


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--converted-model-path", type=str, required=True)
commandline_args = parser.parse_args()


class ProteinBertModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, repr_layers=torch.tensor([])):
        results = model(tokens, repr_layers.tolist())
        results["representations"] = results["representations"][33]
        return results


model, alphabet = load_model_and_alphabet_local(commandline_args.model_path)
model_wrapper = ProteinBertModelWrapper(model)
batch_converter = alphabet.get_batch_converter()

data = [
    ("protein1", "STIEEQAKTFL"),
    ("protein2", "DKFNHEAEDLFYQSS"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    torch.onnx.export(model_wrapper,
        (batch_tokens, torch.tensor([33])),
        commandline_args.converted_model_path,
        use_external_data_format=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": [0, 1]}
    )
