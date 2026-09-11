# Dexbotic DM05 Fast Backend Plugin

This package installs the optional dependencies required by the DM05 TensorRT
and Triton inference backend. It is intended for the CUDA 12.8
`dexmal/dexbotic:dm05` image and leaves the default PyTorch backend unchanged.

The compatible Dexbotic checkout contains the DM05 backend implementation.
This plugin supplies its optional runtime dependencies.

From the mounted Dexbotic checkout inside the container, install the plugin
with:

```bash
bash plugins/dm05-fast/install.sh
```

The plugin installs the ONNX exporter and CUDA 12 TensorRT builder/runtime. It
reuses the Triton version installed with the image's PyTorch wheel. The plugin
source wheel is small, but its first installation downloads approximately 3 GB
of TensorRT CUDA 12 libraries on x86-64.
