import torch
import subprocess
from typing import Optional

def cleanup_gpu(device: Optional[int] = None) -> None:
    """Clears GPU cache and prints memory summary"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
    else:
        print("CUDA not available - no GPU cleanup performed")

def kill_nvidia_processes() -> None:
    """Force kills all NVIDIA processes (Linux/Mac only)"""
    try:
        if not torch.cuda.is_available():
            print("No NVIDIA GPU detected")
            return
            
        # Safer process killing command
        command = (
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader | "
            "xargs -r kill -9 2>/dev/null"
        )
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("Successfully terminated NVIDIA processes")
        else:
            print(f"Failed to kill processes: {result.stderr}")
    except Exception as e:
        print(f"Error killing processes: {e}")