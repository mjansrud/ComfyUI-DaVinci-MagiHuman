"""
Quantize daVinci-MagiHuman model weights to FP8 and save.
Run once, then load the FP8 model directly.

Usage:
    python quantize.py <model_dir> [output_dir]

Example:
    python quantize.py models/daVinci-MagiHuman/distill models/daVinci-MagiHuman/distill_fp8
"""

import sys
import os
import torch
import json
import time
from safetensors.torch import load_file, save_file

def quantize_to_fp8(model_dir: str, output_dir: str):
    """Quantize all safetensor shards to FP8 with per-tensor scales."""

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Group by shard
    shard_to_keys = {}
    for key, shard_file in index["weight_map"].items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    new_weight_map = {}
    scales = {}
    total_orig = 0
    total_fp8 = 0

    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            print(f"  Skipping {shard_file} (not found)")
            continue

        print(f"  Quantizing {shard_file}...")
        t0 = time.time()
        shard_data = load_file(shard_path, device="cpu")

        out_tensors = {}
        for key in shard_to_keys[shard_file]:
            if key not in shard_data:
                continue

            tensor = shard_data[key]
            orig_bytes = tensor.numel() * tensor.element_size()
            total_orig += orig_bytes

            # Only quantize 2D weight tensors (linear layers) in block.*
            # Keep adapter, final_norm, final_linear, rope in original precision
            if tensor.dim() == 2 and "block." in key and tensor.numel() > 1024:
                w = tensor.float()
                amax = w.abs().max().clamp(min=1e-12)
                scale = 448.0 / amax  # FP8 e4m3fn max = 448
                w_fp8 = (w * scale).to(torch.float8_e4m3fn)

                out_tensors[key] = w_fp8
                scales[key] = (1.0 / scale).item()

                fp8_bytes = w_fp8.numel() * w_fp8.element_size()
                total_fp8 += fp8_bytes
            else:
                # Keep as-is (norms, biases, embeddings, adapter, etc.)
                out_tensors[key] = tensor
                total_fp8 += tensor.numel() * tensor.element_size()

            new_weight_map[key] = shard_file

        # Save quantized shard
        out_path = os.path.join(output_dir, shard_file)
        save_file(out_tensors, out_path)
        del shard_data, out_tensors
        print(f"    Saved {out_path} ({time.time()-t0:.1f}s)")

    # Save scale factors
    scales_path = os.path.join(output_dir, "fp8_scales.json")
    with open(scales_path, "w") as f:
        json.dump(scales, f)
    print(f"  Saved {len(scales)} scale factors to {scales_path}")

    # Save updated index
    new_index = {
        "metadata": {
            "total_size": index["metadata"]["total_size"],
            "quantization": "fp8_e4m3fn",
            "original_dir": os.path.abspath(model_dir),
        },
        "weight_map": new_weight_map,
    }
    out_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(out_index_path, "w") as f:
        json.dump(new_index, f, indent=2)

    ratio = total_fp8 / total_orig if total_orig > 0 else 1.0
    print(f"\n  Original: {total_orig / 1e9:.1f} GB")
    print(f"  FP8:      {total_fp8 / 1e9:.1f} GB")
    print(f"  Ratio:    {ratio:.1%}")

    # Merge all shards into a single file
    print(f"\n  Merging shards into single file...")
    all_tensors = {}
    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(output_dir, shard_file)
        if os.path.exists(shard_path):
            shard_data = load_file(shard_path, device="cpu")
            all_tensors.update(shard_data)
            del shard_data

    single_path = os.path.join(output_dir, "model.safetensors")
    save_file(all_tensors, single_path)
    single_size = os.path.getsize(single_path) / 1e9
    print(f"  Saved single file: {single_path} ({single_size:.1f} GB)")

    # Clean up shards
    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(output_dir, shard_file)
        if os.path.exists(shard_path):
            os.remove(shard_path)
    # Remove the index file (no longer needed for single file)
    idx_path = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        os.remove(idx_path)

    del all_tensors
    print(f"  Cleaned up shards.")
    print(f"  Done! Output: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quantize.py <model_dir> [output_dir]")
        print("Example: python quantize.py models/daVinci-MagiHuman/distill models/daVinci-MagiHuman/distill_fp8")
        sys.exit(1)

    model_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = model_dir + "_fp8"

    print(f"Quantizing {model_dir} -> {output_dir}")
    quantize_to_fp8(model_dir, output_dir)
