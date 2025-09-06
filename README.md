# DNN Accelerator for Digit Recognition

This project implements a **Deep Neural Network (DNN) accelerator** on FPGA for handwritten digit recognition using the **MNIST dataset**. It combines **SystemVerilog RTL design** for hardware acceleration with **Python-based training and weight export**.

---

## 📂 Repository Structure

```
DNN-Accelerator-for-Digit-Recognition/
│── Design_files/          # RTL modules in SystemVerilog
│   ├── argmax.sv          # Argmax module for final classification
│   ├── matvec.sv          # Matrix-vector multiplication
│   ├── mnist_accel.sv     # Top-level accelerator design
│   ├── relu.sv            # ReLU activation function
│   ├── rom_ne.sv          # ROM for neuron weights
│
│── mem_files/             # Pre-trained weights and biases (exported from Python)
│   ├── B1.mem, B2.mem     # Bias memory files
│   ├── W2.mem             # Second layer weights
│   ├── image_0000.mem     # Sample input image
│   └── W1/                # First layer neuron-wise weights
│       ├── W1_neuron_0.mem ... W1_neuron_127.mem
│       └── W1.mem         # Consolidated weights
│   └── splitter.py        # Utility script to split/export weights
│
│── model/                 # Python model training and quantization
│   ├── model_train.py     # Training script (MNIST dataset, fully connected DNN)
│   └── mnist_q8_24_export/
│       ├── img/           # Sample test images in hex format
│       └── mem_exp/       # Exported weights/biases for FPGA
│
│── testbench_file/        # FPGA testbench
│   └── tbm.sv
│
│── mnist_FPGA_report.pdf  # Project documentation/report
```

---

## 🚀 Features

- Fully custom **DNN hardware accelerator** for MNIST digit recognition.
- Implemented in **SystemVerilog** with modular design:
  - `matvec.sv`: Efficient matrix-vector multiplication.
  - `relu.sv`: ReLU activation unit.
  - `argmax.sv`: Classification stage.
- Pre-trained weights and biases stored in **memory files (.mem)**.
- Python training script (`model_train.py`) with **quantization support**.
- End-to-end workflow: Train → Export Weights → Load into FPGA → Run Inference.

---

## ⚙️ Workflow

1. **Train the DNN model**
   ```bash
   cd model
   python model_train.py
   ```
   - Trains on MNIST dataset using fully connected layers.
   - Exports weights, biases, and test images to `.mem` files.

2. **Load weights into FPGA**
   - Use files from `mem_files/` (or `model/mnist_q8_24_export/mem_exp/`) in RTL design.

3. **Run simulation**
   ```bash
   cd testbench_file
   # Run testbench (example with QuestaSim or Vivado)
   vsim tbm.sv
   ```

4. **Synthesize on FPGA**
   - Use `mnist_accel.sv` as the top module.
   - Configure memory initialization with `.mem` files.

---

## 🏗️ DNN Accelerator Block Diagram

```
          +-------------------------+
          |      Input Image        |
          |   (28x28 → Flattened)   |
          +-----------+-------------+
                      |
                      v
          +-------------------------+
          |   Fully Connected Layer |
          |        (Weights W1)     |
          +-----------+-------------+
                      |
                      v
          +-------------------------+
          |       ReLU Activation   |
          +-----------+-------------+
                      |
                      v
          +-------------------------+
          |   Fully Connected Layer |
          |        (Weights W2)     |
          +-----------+-------------+
                      |
                      v
          +-------------------------+
          |        Argmax Unit      |
          |   (Selects Final Digit) |
          +-----------+-------------+
                      |
                      v
          +-------------------------+
          |    Predicted Digit      |
          +-------------------------+
```

---

## 🖼️ Example Test Images

Sample input images are provided in:
```
model/mnist_q8_24_export/img/
```
Each `.hex` and `.mem` file corresponds to a digit from the MNIST dataset.

---

## 📖 Documentation

For detailed methodology, design choices, and results, refer to:
```
mnist_FPGA_report.pdf
```

---

## 🔧 Tools & Requirements

- **Python 3.x** with:
  - `numpy`, `torch`, `tensorflow/keras` (for training)
- **SystemVerilog simulator** (e.g., ModelSim, QuestaSim, or Vivado Simulator)
- **FPGA Board** (tested with Xilinx devices, adaptable to others)

---

## 📌 Future Improvements

- Add support for deeper DNN layers and more neurons.
- Explore hardware support for other activations (e.g., sigmoid, tanh).
- Optimize memory usage with BRAM partitioning.
- Extend to datasets beyond MNIST.

---

## 👨‍💻 Authors

Developed as part of a hardware accelerator project for efficient digit recognition using FPGA.
