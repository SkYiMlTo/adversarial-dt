# Neural Attack & Defense Implementation Walkthrough

The project has been successfully expanded to incorporate the **Artificial Intelligence (AI)** component you requested, combining Proposal 1 (Attacking a Neural Defense) and Proposal 2 (GAN-based Evasion).

This expansion maintains your core focus on Digital Twins while injecting state-of-the-art neural architecture into both the defense mechanism (anomaly detection) and the attack generation (evasion).

## 1. What Was Implemented?

The codebase now includes **10 new or modified components**:

### 🛡️ Neural Defense (The Target)
- **LSTM Autoencoder Anomaly Detector** ([`lstm_detector.py`](file:///C:/Users/enzoh/Documents/UQAC/Projets%20Hugo/adversarial-dt/core/lstm_detector.py)): A sequence-to-sequence model that learns the nominal physics noise distribution from the EKF innovations. If an attack introduces unnatural temporal patterns, the LSTM reconstruction error spikes, triggering an alarm.
- **Combined Alarm System** ([`iswt.py`](file:///C:/Users/enzoh/Documents/UQAC/Projets%20Hugo/adversarial-dt/core/iswt.py)): The detection logic was updated so the system now alarms if *any* of the three detectors flag an anomaly: $a(t) = a^{CUSUM} \lor a^{IWD} \lor a^{LSTM}$.

### ⚔️ Neural Attack Generation (The Adversary)
- **Extended Targeted Consistency Attack (TCA)** ([`tca.py`](file:///C:/Users/enzoh/Documents/UQAC/Projets%20Hugo/adversarial-dt/core/tca.py)): We upgraded the PGD attack to compute gradients through the EKF *and* the LSTM simultaneously. The adversary now attempts to keep the LSTM reconstruction error below its threshold while breaking the physical system.
- **GAN Evasion Generator** ([`gan_evasion.py`](file:///C:/Users/enzoh/Documents/UQAC/Projets%20Hugo/adversarial-dt/core/gan_evasion.py)): A Conditional GAN trained to output stealthy sensor perturbations $\delta(t)$ in a single forward pass ($O(1)$) instead of iterative optimization ($O(K \cdot T)$).
- **Physics-Informed Discriminator**: Rather than learning arbitrary visual features, the GAN discriminator literally wraps the `DifferentiableEKF`, CUSUM, and LSTM pipelines to penalize the generator based on true detection statistics.

### 🧪 Experiments & Emulation
- **S3 Experiment Script** ([`run_s3_neural.py`](file:///C:/Users/enzoh/Documents/UQAC/Projets%20Hugo/adversarial-dt/experiments/run_s3_neural.py)): A brand new evaluation suite that generates Tables 6 (LSTM Baseline), 7 (Adversarial LSTM Robustness), and 8 (GAN vs PGD performance).
- **Docker Integration** ([`main.py`](file:///C:/Users/enzoh/Documents/UQAC/Projets%20Hugo/adversarial-dt/services/digital_twin/main.py)): I updated the Digital Twin microservice to dynamically load the LSTM neural components if PyTorch is available inside the container, answering your question about Docker usability!

> [!TIP]
> The conditional PyTorch imports mean the core non-neural code will continue to function on environments where PyTorch is not available, keeping the foundational project pristine.

## 2. Validation Results

A rigorous testing suite was executed in the `.venv` virtual environment to ensure the mathematical validity of the code:

1. **14 Existing Tests Passed**: The legacy EKF, CUSUM, and ISWT tests ran successfully, proving the new code didn't break backward compatibility.
2. **16 New Neural Tests Passed**:
   - `test_lstm.py`: Verified anomaly score computations, thresholding, gradient flow for the attacker, and that the LSTM correctly flags biased attacks.
   - `test_gan.py`: Verified the $L_\infty$ budget constraints ($\delta_i \leq \epsilon_i$), the discriminator's output range, and the GAN end-to-end training loop.

*Note: One minor tensor shape mismatch was found during the GAN integration tests and immediately hotfixed.*

## 3. How to Generate Your Paper Results

You are now ready to generate the academic output. From the root of your project, run:

```bash
# To run the new neural experiments (generates Tables 6, 7, 8)
make run-s3

# To generate the updated PDF plots (including new GAN dynamics and LSTM ROCs)
make figures
```

The output will be dumped securely in your `results/s3/` directory!
