"""
Hunyuan3D 2.1 — extension setup script (AMD/DirectML fork by Irongate3D).

Adds AMD GPU support via torch-directml on Windows.
On NVIDIA, behaviour is identical to upstream Mini extension.

Original: https://github.com/lightningpixel/modly-hunyuan3d-mini-extension
Fork:     https://github.com/Irongate3D/modly-hunyuan3d-21-extension

Called by Modly at extension install time with:
    python setup.py <json_args>
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def detect_amd_gpu() -> bool:
    """Returns True if AMD GPU present and no NVIDIA driver found."""
    if platform.system() != "Windows":
        return False
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.upper()
        return ("AMD" in output or "RADEON" in output) and "NVIDIA" not in output
    except Exception:
        return False


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int,
    cuda_version:  int = 0,
    torch_flavor:  str = "cuda",
    accelerator:   str = "",
    platform_name: str = "",
) -> None:
    venv    = ext_dir / "venv"
    is_win  = platform.system() == "Windows"
    is_mac  = platform.system() == "Darwin" or platform_name == "darwin"
    is_amd  = detect_amd_gpu()

    if not accelerator:
        if is_mac:
            accelerator = "mps" if platform.machine().lower() == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print(f"[setup] accelerator={accelerator}  gpu_sm={gpu_sm}  cuda_version={cuda_version}  amd={is_amd}")
    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if is_mac:
        print("[setup] macOS -> PyTorch from standard PyPI")
        pip(venv, "install", "torch", "torchvision")

    elif is_amd and is_win:
        # AMD on Windows — CPU PyTorch + torch-directml
        print("[setup] AMD GPU detected on Windows — installing CPU PyTorch + torch-directml …")
        pip(venv, "install",
            "torch==2.4.1",
            "torchvision==0.19.1",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        )
        pip(venv, "install", "torch-directml")
        print("[setup] torch-directml installed — AMD GPU acceleration enabled via DirectML.")

    elif torch_flavor == "rocm":
        # ROCm on Linux
        print("[setup] -> PyTorch + ROCm 7.2")
        pip(venv, "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/rocm7.2")

    elif gpu_sm >= 100 or cuda_version >= 128:
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8")
        pip(venv, "install", "torch==2.7.0", "torchvision==0.22.0",
            "--index-url", "https://download.pytorch.org/whl/cu128")

    elif gpu_sm >= 70:
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
        pip(venv, "install", "torch==2.6.0", "torchvision==0.21.0",
            "--index-url", "https://download.pytorch.org/whl/cu124")

    else:
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")
        pip(venv, "install", "torch==2.5.1", "torchvision==0.20.1",
            "--index-url", "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies …")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "trimesh",
        "pymeshlab",
        "opencv-python-headless",
        "huggingface_hub",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "einops",
        "scipy",
        "scikit-image",
        "safetensors",
    )

    # ------------------------------------------------------------------ #
    # rembg — use CPU onnxruntime on AMD (DirectML not supported by rembg)
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg …")
    if is_mac or torch_flavor == "rocm" or is_amd:
        pip(venv, "install", "rembg", "onnxruntime")
        print("[setup] rembg installed with CPU onnxruntime.")
    elif gpu_sm >= 70:
        pip(venv, "install", "rembg[gpu]")
    else:
        pip(venv, "install", "rembg", "onnxruntime")

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            torch_flavor = sys.argv[5] if len(sys.argv) >= 6 else "cuda",
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe    = args["python_exe"],
            ext_dir       = Path(args["ext_dir"]),
            gpu_sm        = int(args["gpu_sm"]),
            cuda_version  = int(args.get("cuda_version", 0)),
            torch_flavor  = args.get("torch_flavor", "cuda"),
            accelerator   = args.get("accelerator", ""),
            platform_name = args.get("platform", ""),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
