import deepspeed
import torch

def optimize_memory(model, device):
    if device.type == "cuda":
        # Configure DeepSpeed for memory optimization
        ds_config = {
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_fp16_weights_on_model_save": True
            }
        }

        # Initialize DeepSpeed engine
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        return model_engine, optimizer
    else:
        print("Running on CPU. Memory optimization skipped.")
        return model, None
