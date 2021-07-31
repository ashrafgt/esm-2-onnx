## FAIR ESM to ONNX:

### Overview:

Some of [Facebook AI Research's ESM models](https://github.com/facebookresearch/esm) are fairly large. One option for speeding up the prediction is converting the Pytorch model to ONNX and using [ONNX Runtime](https://github.com/microsoft/onnxruntime) to run the inference.


### Motivation:

While the conversion should be possible just using the familiar `torch.onnx.export()` syntax, if you want to extract embeddings from the model, you'll most likely want to pass the `repr_layers` argument to the model's `forward()` method. 

Trying to pass this argument inside the `export()` call will cause an exception (because while `ProteinBertModel.forward()` expects a plain-old `[]`, a JIT-compatible format such as `torch.tensor([])` is required for the export operation).


### Walkthrough:

#### 1. Setup the environment:

Option 1: Using Docker
```bash
docker build -t esm-2-onnx -f Dockerfile.CUDA .  # if you have a CUDA-capable machine with nvidia-container-runtime
docker run -it --gpus=0 esm-2-onnx:latest bash  # running using just 1 GPU (also set inside the Dockerfile)
# or
docker build -t esm-2-onnx -f Dockerfile.CPU .  # otherwise
docker run -it esm-2-onnx:latest bash
```
> Consider adding a bind-mount argument to the `docker run` command in order to persist the model files: `-v /mnt/models/esm1b:/mnt/models/esm1b`

Option 2: Without using Docker
```bash
pip install -r requirements.CUDA.txt  # if you have a CUDA-capable machine
# or
pip install -r requirements.CPU.txt  # otherwise
```

#### 2. Download the model checkpoint:
```bash
MODEL_DIRECTORY_PATH=/mnt/models/esm1b
mkdir -p $MODEL_DIRECTORY_PATH
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt -P $MODEL_DIRECTORY_PATH
```

#### 3. Create directories for the converted and CLI-optimized ONNX graph files:
```bash
MODEL_PATH=$MODEL_DIRECTORY_PATH/esm1b_t33_650M_UR50S.pt
CONVERTED_GRAPH_PATH=$MODEL_DIRECTORY_PATH/converted/esm1b_t33_650M_UR50S_graph.onnx
OPTIMIZED_GRAPH_PATH=$MODEL_DIRECTORY_PATH/optimized/esm1b_t33_650M_UR50S_graph.onnx
mkdir -p $(dirname $CONVERTED_GRAPH_PATH) $(dirname $OPTIMIZED_GRAPH_PATH)
```

#### 4. Convert the model using `torch.onnx` and further optimize it using `onnxruntime_tools`:
```bash
python src/convert.py --model-path $MODEL_PATH --converted-model-path $CONVERTED_GRAPH_PATH
python -m onnxruntime_tools.optimizer_cli --float16 --opt_level 99 --use_gpu \
   --model_type bert --hidden_size 1024 --num_heads 16 --input $CONVERTED_GRAPH_PATH \
   --output $OPTIMIZED_GRAPH_PATH  # convert to float 16 precision and apply all available optimizations
```

#### 5. Test the prediction using `onnx_runtime.InferenceSession()`:
```bash
python src/predict.py --model-path $OPTIMIZED_GRAPH_PATH
```
