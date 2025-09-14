# SPGR
DNN Repair Method
# SPGR: Source-Path Guided Repair for Neural Networks

This repository provides implementations of two repair tasks for deep neural networks: **safety repair on ACAS Xu networks** and **robustness repair on MNIST models**.  
Both tasks aim to identify and repair faulty neurons with minimal parameter modifications, improving model safety and robustness while preserving correct behaviors.

---

##  Tasks

### 1. ACAS Xu Safety Repair
- **Entry file:** `ACASXU_Repir.py`
- Description: Repair erroneous behaviors in the ACAS Xu benchmark networks under safety-critical constraints.
- ### Run ACAS Xu safety repair：
python ACASXU_Repir.py --alpha 0.5 --beta 0.1 --lambda1 0.9 --lambda2 0.1

### 2. MNIST Robustness Repair
- **Entry file:** `MNIST_Repair.py`
- Description: Enhance the robustness of MNIST classification models against counterexamples and adversarial-like perturbations.
- ### Run MNIST robustness repair：
python MNIST_Repair.py --alpha 0.5 --beta 0.3 --lambda1 0.9 --lambda2 0.1

---

##  Acknowledgements
This repository is built upon and inspired by the following excellent works:
- [BIRDNN](https://github.com/ByteTao5/BIRDNN)  
- [CARE](https://github.com/sunbingsmu/care)  

We thank the authors for their contributions.

---

##  Arguments

Both tasks support the following parameters:

```python
parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off factor between safety and accuracy')
parser.add_argument('--beta', type=float, default=0.1, help='Weight factor for path repair optimization')
parser.add_argument('--lambda1', type=float, default=0.9, help='Source neuron scoring parameter λ1')
parser.add_argument('--lambda2', type=float, default=0.1, help='Source neuron scoring parameter λ2')





