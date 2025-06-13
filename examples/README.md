# ACT Policy Inference

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face datasets
- LeRobot package
- huggingface_hub (for downloading model configs)

## Usage

### Basic Usage (Default Model)

```bash
python inference_test.py
```
This will:
- Load the default model (jesbu1/act-bridge-v2)
- Use the bridge_v2_lerobot dataset
- Run inference on the first sample

### Command Line Arguments

#### Model Selection
```bash
# Use a different Hugging Face model
python inference_test.py --hf_model "username/model-name"

# Use a local checkpoint
python inference_test.py --local_checkpoint "/path/to/checkpoint"

# Use a local checkpoint with custom config
python inference_test.py --local_checkpoint "/path/to/checkpoint" --config_path "/path/to/train_config.json"
```

#### Dataset Options
```bash
# Use a different dataset
python inference_test.py --dataset "other/dataset"

# Use a different split
python inference_test.py --split "test"

# Use a specific sample
python inference_test.py --sample_idx 5
```

### Complete Example

```bash
python inference_test.py \
    --local_checkpoint "/home/user/checkpoints/act_model" \
    --config_path "/home/user/checkpoints/train_config.json" \
    --dataset "custom/dataset" \
    --split "validation" \
    --sample_idx 10
```

## Model Requirements

### Hugging Face Models
- Must be compatible with the ACT policy architecture
- Should contain:
  - Model weights (automatically downloaded)
  - `train_config.json` (automatically downloaded)
  - Required fields in config:
    - `policy.input_features`: List of input feature configurations
    - `policy.output_features`: List of output feature configurations
    - `policy.device`: Device to run inference on
    - `policy.chunk_size`: Size of action chunks
    - `policy.n_action_steps`: Number of action steps

### Local Checkpoints
Should contain:
- `model.safetensors` or `pytorch_model.bin` (model weights)
- `train_config.json` (training configuration) with the same required fields as above

## Config File Format

The training config file should be a JSON file with the following structure:
```json
{
  "policy": {
    "input_features": [
      {
        "type": "state",
        "shape": [7]
      }
    ],
    "output_features": [
      {
        "type": "action",
        "shape": [8]
      }
    ],
    "device": "cuda",
    "chunk_size": 100,
    "n_action_steps": 100
  }
}
```

## Output

The script provides detailed logging of:
- Dataset loading progress
- Sample information
- State shapes and dimensions
- Model loading status
- Inference results

Example output:
```
[INFO] Loading dataset jesbu1/bridge_v2_lerobot...
[INFO] Dataset loaded. Number of samples: 1999410
[INFO] Sample keys: dict_keys(['observation.state', 'action', 'camera_present', ...])
[INFO] Extracting state from sample...
[INFO] State extracted. Shape: (7,)
[INFO] Loading model from Hugging Face Hub: jesbu1/act-bridge-v2
[INFO] Downloading config file for jesbu1/act-bridge-v2...
[INFO] Model loaded successfully.
[INFO] Running inference...
[INFO] Inference complete.
State shape: (7,)
Predicted action shape: (8,)
Predicted action: [0.1, 0.2, ...]
```

## Error Handling

Common errors and solutions:
1. `train_config.json not found`: 
   - For local checkpoints: Specify the correct path using `--config_path`
   - For Hugging Face models: The script will attempt to download the config file
2. `Invalid model ID`: Check the Hugging Face model ID
3. `Dataset not found`: Verify the dataset ID and your internet connection
4. `Missing required config fields`: Ensure your train_config.json has all required fields
5. `Model architecture mismatch`: Verify the model is compatible with ACT policy

## Contributing

Feel free to:
- Add support for more model architectures
- Implement batch inference
- Add visualization tools
- Improve error handling
- Add support for more config file formats

## License

This code is part of the LeRobot project. See the main repository for license information.