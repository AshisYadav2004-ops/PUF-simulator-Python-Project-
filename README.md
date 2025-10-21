# 🧠 PUF Simulator & Neural Network Model

This project simulates **Arbiter PUFs (APUFs)** and **XOR-APUFs** in Python and trains a **Neural Network** using TensorFlow to model their behavior.
It’s designed for research, learning, and testing ML-based PUF modeling attacks.

---

## ⚙️ Features

* ✅ **Arbiter PUF (APUF)** implementation using additive delay model
* ✅ **XOR-APUF** support (combine multiple APUFs)
* ✅ Generate Challenge–Response Pairs (CRPs)
* ✅ Add configurable noise to simulate environment variation
* ✅ **Neural Network (TensorFlow)** trainer to model PUF behavior
* ✅ Example experiments with results and visualization

---

## 📁 Project Structure

```
puf_project/
│
├── puf_simulator.py        # Core Arbiter and XOR PUF implementation
├── nn_puf_model.py         # TensorFlow model for learning PUF mapping
├── requirements.txt        # Required dependencies
└── README.md               # This file
```

---

## 🧩 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/puf-simulator.git
cd puf-simulator
```

### 2️⃣ Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
tensorflow
scikit-learn
```

---

## 🚀 Usage

### Run Arbiter PUF Simulator

```bash
python puf_simulator.py
```

This runs the demo inside `puf_simulator.py` which:

* Generates random challenges and PUF responses
* Tests stability under noise
* Prints statistical summary

### Run Neural Network Model

```bash
python nn_puf_model.py
```

This script:

* Creates CRPs using the Arbiter or XOR-APUF
* Builds a TensorFlow neural network
* Trains the NN to learn the PUF mapping
* Prints training and test accuracy

---

## 🧠 How It Works

### Arbiter PUF (APUF)

* Takes binary challenge bits.
* Each bit controls signal routing between two delay paths.
* The arbiter outputs `1` or `0` based on which path arrives first.

### XOR-APUF

* Multiple APUFs are evaluated in parallel.
* Their outputs are XORed → final response.
* Increases modeling resistance.

### Neural Network Modeling

* Uses CRP data to train an MLP (feedforward neural network).
* Predicts PUF responses from unseen challenges.
* High accuracy on APUF → demonstrates learnability.
* Lower accuracy on XOR-APUF → demonstrates resistance.

---

## 📊 Example Output

```
=== APUF simulator demo ===
APUF: generated 10000 CRPs, 0.512 fraction of ones
APUF: noise-induced response flip fraction (noise_std=0.2): 0.031
Final test accuracy: 0.9982
Generalization accuracy on unseen CRPs: 0.9975
```

---

## 🧩 Customize

You can easily modify:

* Challenge length `k`
* XOR degree `m`
* Neural network size (hidden layers, neurons)
* Noise level

In `nn_puf_model.py`, edit:

```python
k = 16
xor_degree = 3
hidden_units = [128, 64]
epochs = 30
```

---

## 📚 References

* Suh & Devadas, *"Physical Unclonable Functions for Device Authentication and Secret Key Generation"*, DAC 2007.
* Lim et al., *"A Statistical Modeling Attack on the XOR Arbiter PUF"*, CHES 2005.
* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

---

## 🧑‍💻 Author

**Ashis**

> Exploring Hardware Security and AI-driven PUF modeling.

---

## 🪪 License

This project is released under the **MIT License**.
Feel free to use and modify it for research and educational purposes.
