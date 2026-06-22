"""
Hunyuan3D 2.1 — Modly extension (AMD/DirectML fork by Irongate3D).

Shape-only generation (no texture) — pure PyTorch, no CUDA compiled extensions.
Runs on AMD via torch-directml on Windows, or CUDA on NVIDIA.

Original Mini extension: https://github.com/lightningpixel/modly-hunyuan3d-mini-extension
Model: https://huggingface.co/tencent/Hunyuan3D-2
Fork:  https://github.com/Irongate3D/modly-hunyuan3d-21-extension
"""
import io
import os
import random
import sys
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID   = "tencent/Hunyuan3D-2"
_SUBFOLDER    = "hunyuan3d-dit-v2-0"
_GITHUB_ZIP   = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip"


def _get_device():
    """
    Resolve best available compute device.
    Priority: CUDA > DirectML (AMD on Windows) > CPU
    """
    import torch
    if torch.cuda.is_available():
        print("[Hunyuan3D21Generator] Using CUDA.")
        return "cuda", torch.float16

    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"[Hunyuan3D21Generator] Using DirectML: {torch_directml.device_name(0)}")
        return dml_device, torch.float16
    except ImportError:
        pass

    print("[Hunyuan3D21Generator] WARNING: No GPU acceleration — using CPU.")
    import torch
    return "cpu", torch.float32


class Hunyuan3D21Generator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-21"
    DISPLAY_NAME = "Hunyuan3D 2.1"
    VRAM_GB      = 10

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        model_dir = self.model_dir / _SUBFOLDER
        return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen()

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        if sys.platform == "darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float32
        else:
            device, dtype = _get_device()

        print(f"[Hunyuan3D21Generator] Loading pipeline from {self.model_dir} (subfolder={_SUBFOLDER})…")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=_SUBFOLDER,
            use_safetensors=True,
            device=device,
            dtype=dtype,
        )
        self._model  = pipeline
        self._device = device
        self._dtype  = dtype
        print(f"[Hunyuan3D21Generator] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif sys.platform == "darwin" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps      = int(params.get("num_inference_steps", 30))
        vert_count     = int(params.get("vertex_count", 0))
        octree_res     = int(params.get("octree_resolution", 512))
        guidance_scale = float(params.get("guidance_scale", 5.5))
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 12, "Generating 3D shape…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, 82, "Generating 3D shape…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
            self._report(progress_cb, 85, "Optimizing mesh…")
            mesh = self._decimate(mesh, vert_count)

        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[Hunyuan3D21Generator] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3D21Generator] Downloading {_HF_REPO_ID}…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-0-fast/**",
                "hunyuan3d-dit-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0-withencoder/**",
                "hunyuan3d-paint-v2-0/**",
                "hunyuan3d-paint-v2-0-turbo/**",
                "hunyuan3d-delight-v2-0/**",
                "assets/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )
        print("[Hunyuan3D21Generator] Download complete.")

    def _ensure_hy3dgen(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dgen still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_hy3dgen(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D21Generator] Downloading hy3dgen source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        print("[Hunyuan3D21Generator] Extracting hy3dgen…")

        prefix = "Hunyuan3D-2-main/hy3dgen/"
        strip  = "Hunyuan3D-2-main/"

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Hunyuan3D21Generator] hy3dgen extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Quality",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast"},
                    {"value": 30, "label": "Balanced"},
                    {"value": 50, "label": "High"},
                ],
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 512,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 380, "label": "Medium"},
                    {"value": 512, "label": "High"},
                    {"value": 768, "label": "Ultra (slow)"},
                ],
                "tooltip": "Octree resolution. Higher = more detail but slower and more VRAM. 512 recommended for 3D printing.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "float",
                "default": 5.5,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input image.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     4294967295,
                "tooltip": "Seed for reproducibility. Set to -1 for a random seed.",
            },
        ]
