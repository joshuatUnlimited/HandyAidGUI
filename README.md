# HandyAidGUI Vulkan explicit-device fix

Replace these files on GitHub:

- `app/engine/command_builder.py`
- `app/backends/gpu_manager.py`
- `app/backends/vulkan.py`

## Important behavior change

`GGML_VK_VISIBLE_DEVICES` and `transcribe-cli --device` are separate controls.

The GUI previously exposed GPUs through `GGML_VK_VISIBLE_DEVICES` but did not explicitly select the compute device. This revision adds `--device 0` for the process-local Vulkan device.

For multi-GPU workers, each worker exposes exactly one physical Vulkan GPU and therefore uses process-local device `0`. This explicitly pins:

- worker 0 -> selected physical Vulkan GPU 0
- worker 1 -> selected physical Vulkan GPU 1

The GPU manager also probes each physical adapter individually, rejects adapters that do not appear to have enough free VRAM for a full model copy, and distributes audio by available VRAM.

This is process-level balancing, not pooled VRAM or tensor/model splitting.
