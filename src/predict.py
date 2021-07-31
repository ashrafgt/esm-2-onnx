import argparse
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_device,
)
from esm.data import Alphabet


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
commandline_args = parser.parse_args()


options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
model = InferenceSession(commandline_args.model_path)
provider = "CUDAExecutionProvider" if get_device() == "GPU" else "CPUExecutionProvider"
model.set_providers([provider])
alphabet = Alphabet.from_architecture("protein_bert_base")
batch_converter = alphabet.get_batch_converter()

sequences = ["STIEEQAKTFL", "DKFNHEAEDLFYQSS"]

data = [(str(i), sequence) for i, sequence in enumerate(sequences)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
output = model.run(None, {"inputs": batch_tokens.numpy()})
embeddings = output[1][:, 0]

print(embeddings)
