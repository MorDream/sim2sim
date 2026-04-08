import torch
import argparse
import os
from collections import OrderedDict

def inspect_pt_model(model_path, verbose=False, max_depth=3):
    """
    Inspect the structure of a PyTorch .pt model file, handling multiple formats.
    
    Args:
        model_path: Path to the .pt file
        verbose: Whether to show more detailed information
        max_depth: Maximum depth to print nested structures
    """
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} does not exist.")
        return
    
    print(f"=" * 70)
    print(f"Inspecting model file: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"=" * 70)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"\n[1] Top-level type: {type(checkpoint).__name__}")
    
    if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
        print(f"[2] Number of top-level keys: {len(checkpoint)}")
        print(f"[3] Top-level keys: {list(checkpoint.keys())}")
        print("\n" + "-" * 70)
        
        for key, value in checkpoint.items():
            print(f"\n🔹 Key: '{key}'")
            print(f"   ├─ Type: {type(value).__name__}")
            
            if isinstance(value, OrderedDict) or isinstance(value, dict):
                print(f"   ├─ Number of entries: {len(value)}")
                if verbose and len(value) > 0:
                    sample_keys = list(value.keys())[:10]
                    print(f"   ├─ First {min(10, len(value))} keys: {sample_keys}")
                    if len(value) > 10:
                        print(f"   └─ ... and {len(value) - 10} more")
                # Print some example layers
                if verbose and len(value) > 0:
                    print(f"\n   📋 Sample layers:")
                    for i, (layer_name, layer_data) in enumerate(list(value.items())[:5]):
                        if hasattr(layer_data, 'shape'):
                            print(f"      {layer_name}: {layer_data.shape}")
                        else:
                            print(f"      {layer_name}: {type(layer_data).__name__}")
                    if len(value) > 5:
                        print(f"      ... and {len(value) - 5} more layers")
            elif hasattr(value, 'shape'):
                print(f"   └─ Shape: {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"   └─ Length: {len(value)}")
            else:
                print(f"   └─ Value: {value}")
    
    elif isinstance(checkpoint, torch.nn.Module):
        print(f"[2] Model is a full torch.nn.Module")
        print(f"[3] Model class: {checkpoint.__class__.__name__}")
        print(f"[4] Number of parameters: {sum(p.numel() for p in checkpoint.parameters()):,}")
        print("\nModel state_dict keys (first 20):")
        state_dict = checkpoint.state_dict()
        for i, (name, param) in enumerate(list(state_dict.items())[:20]):
            print(f"  {name}: {param.shape}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")
    
    else:
        print(f"[2] Unexpected format: {type(checkpoint)}")
        print(f"[3] Content preview: {checkpoint}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("-" * 70)
    
    analyze_structure(checkpoint, verbose)
    
    return checkpoint

def analyze_structure(checkpoint, verbose=False):
    """Analyze what kind of checkpoint this is"""
    
    # Case 1: This is already a state_dict (common case)
    if isinstance(checkpoint, (OrderedDict, dict)) and all(isinstance(v, torch.Tensor) for v in list(checkpoint.values())[:10]):
        total_params = sum(v.numel() for v in checkpoint.values() if isinstance(v, torch.Tensor))
        print(f"✅ This looks like a **state_dict** (just the model weights)")
        print(f"   Number of layers/tensors: {len(checkpoint)}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Total size: {sum(v.element_size() * v.nelement() for v in checkpoint.values() if isinstance(v, torch.Tensor)) / (1024*1024):.2f} MB")
        return
    
    # Case 2: Standard checkpoint with model_state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        msd = checkpoint['model_state_dict']
        total_params = sum(v.numel() for v in msd.values() if isinstance(v, torch.Tensor))
        print(f"✅ This looks like a **full training checkpoint** (contains model_state_dict)")
        print(f"   Number of layers/tensors in model_state_dict: {len(msd)}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Total size: {sum(v.element_size() * v.nelement() for v in msd.values() if isinstance(v, torch.Tensor)) / (1024*1024):.2f} MB")
        # Check for other common keys
        extra_keys = [k for k in checkpoint.keys() if k != 'model_state_dict']
        if extra_keys:
            print(f"   Additional keys in checkpoint: {extra_keys}")
        return
    
    # Case 3: Checkpoint with just 'model' key
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_data = checkpoint['model']
        if isinstance(model_data, (OrderedDict, dict)):
            total_params = sum(v.numel() for v in model_data.values() if isinstance(v, torch.Tensor))
            print(f"✅ This looks like a **checkpoint with 'model' key**")
            print(f"   Number of layers/tensors: {len(model_data)}")
            print(f"   Total parameters: {total_params:,}")
            extra_keys = [k for k in checkpoint.keys() if k != 'model']
            if extra_keys:
                print(f"   Additional keys: {extra_keys}")
            return
    
    # Case 4: Checkpoint with state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
        total_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        print(f"✅ This looks like a **checkpoint with 'state_dict' key** (common in many frameworks)")
        print(f"   Number of layers/tensors: {len(sd)}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Total size: {sum(v.element_size() * v.nelement() for v in sd.values() if isinstance(v, torch.Tensor)) / (1024*1024):.2f} MB")
        extra_keys = [k for k in checkpoint.keys() if k != 'state_dict']
        if extra_keys:
            print(f"   Additional keys in checkpoint: {extra_keys}")
        return
    
    # Case 5: This is a full model saved
    if isinstance(checkpoint, torch.nn.Module):
        total_params = sum(p.numel() for p in checkpoint.parameters())
        print(f"✅ This is a **full serialized model** (torch.nn.Module)")
        print(f"   Model class: {checkpoint.__class__.__name__}")
        print(f"   Total parameters: {total_params:,}")
        return
    
    # Fallback
    print(f"⚠️  Unrecognized format. Type: {type(checkpoint).__name__}")
    if isinstance(checkpoint, dict):
        print(f"   Available keys: {list(checkpoint.keys())}")

def print_all_keys(checkpoint, prefix=''):
    """Recursively print all keys in the checkpoint"""
    if isinstance(checkpoint, (dict, OrderedDict)):
        for key, value in checkpoint.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, OrderedDict)):
                print(full_key)
                print_all_keys(value, full_key)
            else:
                if hasattr(value, 'shape'):
                    print(f"{full_key}: {value.shape}")
                else:
                    print(f"{full_key}: {type(value).__name__}")

def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch .pt model file structure')
    parser.add_argument('path', type=str, help='Path to the .pt model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    parser.add_argument('--all-keys', '-a', action='store_true', help='Print all keys recursively')
    parser.add_argument('--max-depth', '-d', type=int, default=3, help='Maximum depth for nested printing')
    args = parser.parse_args()
    
    checkpoint = inspect_pt_model(args.path, args.verbose, args.max_depth)
    
    if args.all_keys and checkpoint is not None and isinstance(checkpoint, (dict, OrderedDict)):
        print("\n" + "=" * 70)
        print("All keys recursively:")
        print("-" * 70)
        print_all_keys(checkpoint)

if __name__ == '__main__':
    main()
