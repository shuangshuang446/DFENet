## Abstract
Sensor-based human activity recognition (SHAR) has attracted considerable attention due to the development of wearable sensing technology. Despite significant progress in SHAR through deep learning methods, developing a model that can enhance feature representation without adding extra computational burden remains a common issue. On the one hand, convolutional neural networks (CNNs) are proficient at extracting local spatial features, they fall short in capturing the temporal dynamics across multiple sensor modalities. On the other hand, recurrent neural networks (RNNs) are capable of processing sequential information but are hampered by inefficiency. To this end, we propose an efficient Dynamic Feature Enhancement Network (DFENet) for SHAR tasks. Specifically, DFENet incorporates the Sensor Signal Feature Refinement (SSFR) and Dynamic Spatial-Channel Mask (DSCM) modules to deliver accurate action recognition with efficient computation. The SSFR module efficiently removes spatial and channel redundancies in features to extract key action-related information, while the DSCM module dynamically skips unnecessary computations and selectively activates relevant spatial and channel regions. We evaluate DFENet's performance on four public datasets, achieving state-of-the-art accuracy of 96.73\% on UCI-HAR, 98.31\% on WISDM, 93.44\% on PAMAP2, and 76.52\% on UniMiB-SHAR with negligible computational burden, which highlights its efficiency in SHAR tasks.

# **Model Deployment**

This guide explains how to convert our PyTorch model to MNN format, apply FP16 quantization, and deploy it on mobile devices (Android).

---
## **Prerequisites**

1. **Android Device** with USB debugging enabled.
2. **ADB Tool**: Installed and properly configured.
3. **MNN Tools**: For MNN model generation and quantification.

---

## **Step 1: Convert PyTorch Model to ONNX Format**

Export your PyTorch model to ONNX format with the following script:
```bash
python deployment/torch_to_onnx.py
```
---

## **Step 2: Convert ONNX Model to MNN Format**

1. Clone and build the MNN repository:

```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build
cd build
cmake ..
make -j
```

2. Use the `onnx2mnn` tool to convert the ONNX model to MNN format:

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --quantize FP16
```

This command converts `model.onnx` to `model.mnn` with FP16 quantization.

---

## **Step 3: Transfer Files to the Android Device**

1. **Connect the Device**  
   Ensure the Android device is connected and detected by ADB:
   ```bash
   adb devices
   ```

2. **Push Files to the Device**  
   Transfer the model and `mnncli` to the device:
   ```bash
   adb push model.mnn /data/local/tmp/
   adb push mnncli /data/local/tmp/
   adb shell chmod +x /data/local/tmp/mnncli
   ```

---

## **Step 4: Run Runtime Tests**

1. **Access the Device Shell**  
   Open the Android shell:
   ```bash
   adb shell
   ```

2. **Run the Model with Benchmark Script**  
   Navigate to the directory and execute the benchmark test:
   ```bash
   cd /data/local/tmp/
   ./bench_android.sh -p model.mnn
   ```

   - Use `-64` for ARMv8 or other specific options if needed.

3. **Inspect the Output**  
   The results will be saved to a `benchmark.txt` file in the same directory. Example result:
   ```
   Forward type: 4, thread=4
   Input shape: [1, 1, 200, 3]
   Output shape: [1, 6]
   Inference time:
   ModelName,Min,Max,Avg,Std
   dfe.mnn,7686,12252,10171.0,723.006
   ```

---

With these steps, we can deploy the PyTorch model using MNN on mobile devices for simple runtime test.
