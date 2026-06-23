# Digital Twin Security Experimental Framework

This repository contains the complete, reproducible experimental framework for our paper on **Securing Digital Twins against Targeted Consistency Attacks (TCA)** using the **Innovation Whiteness Detector (IWD)**.

It provides both a fast, automated offline evaluation pipeline (for generating the paper's statistical results and figures) and a full Dockerized hardware-in-the-loop emulation of an industrial Control System.

## 📖 What does this framework do?
Industrial Control Systems (ICS) rely on Digital Twins (DT) for intrusion detection. However, advanced attackers can perform a **Targeted Consistency Attack (TCA)**: they inject physical faults (e.g., draining a tank) while simultaneously injecting perfectly crafted fake sensor readings that trick the Digital Twin's standard detectors (like CUSUM) into thinking everything is perfectly normal.

Our defense, the **Innovation Whiteness Detector (IWD)**, uses the **Stein Matrix Divergence** to monitor the spatial cross-correlation of the Digital Twin's internal errors (innovations). Because of fundamental physical constraints, the attacker *must* introduce unnatural correlations across sensors to hide their fault. The IWD catches this mathematically.

This codebase simulates these attacks, tests the defense, and generates the exact graphs and LaTeX tables used in the paper.

---

## 🚀 Quick Start (Generating Paper Results)

You don't need any special hardware to run the experiments. Everything runs locally on your machine.

### Prerequisites
1. Python 3.9+ 
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments
To run the entire evaluation pipeline and generate all tables and figures for the paper, simply use the `make` command:

```bash
make run-all
```

This will automatically:
1. Run the **S1 Automated Evaluation** (thousands of simulated attacks).
2. Run the **S2 SWaT Evaluation** (validation on the Secure Water Treatment dataset).
3. Generate the **LaTeX tables** and **PDF figures**.

### Where are the results?
Once finished, look in the `results/` directory:
- `results/tables/`: Contains `.tex` files that you can copy-paste directly into the paper's LaTeX source.
- `results/figures/`: Contains the `.pdf` graphs (e.g., `sds_convergence.pdf`, `budget_sweep.pdf`) used in the paper.

---

## 📂 Project Structure

The repository is organized logically into core algorithms, experimental scripts, and real-time emulation services.

### 1. Core Algorithms (`core/`)
The mathematical heart of the paper.
- `process_model.py`: Nonlinear ODE simulation of the Two-Tank water distribution process using RK4 integration.
- `ekf.py`: Extended Kalman Filter (EKF) that tracks the state of the physical system. Includes a specialized PyTorch-differentiable EKF for the attacker.
- `cusum.py`: The standard CUSUM anomaly detector (which the attacker tries to evade).
- `iswt.py`: **Our Novel Defense**. The Innovation Spatial Whiteness Test, utilizing Stein divergence and empirical thresholding.
- `tca.py`: The Targeted Consistency Attack. Uses PyTorch's Projected Gradient Descent (PGD) to optimize a highly sophisticated surrogate loss function to evade detection. Includes `run_whitebox_neural()` for evading the combined CUSUM+ISWT+LSTM pipeline.
- `sds.py`: The Sensor Deception Score (SDS) metric that quantifies how successfully the attacker tricked the system.
- `calibration.py`: Pre-session calibration pipeline to baseline the system's nominal noise profile.
- `lstm_detector.py`: **LSTM Autoencoder Anomaly Detector**. Learns the nominal innovation distribution and detects attacks via reconstruction error. Provides a differentiable interface for adversarial gradient attacks.
- `gan_evasion.py`: **Conditional GAN Evasion Generator**. Trains a generator G(z,c) to produce stealthy perturbations in a single forward pass, replacing iterative PGD optimization. Includes a physics-informed discriminator wrapping the full detection pipeline.

### 2. Experiments (`experiments/`)
Scripts to generate the statistical claims made in the paper.
- `run_s1_automated.py`: Runs the White-Box and Grey-Box TCA attacks over 30 random sessions across varying perturbation budgets to populate Tables 1, 3, and 5.
- `run_s2_offline.py`: Validates the defense on the real-world **SWaT (Secure Water Treatment)** dataset (Table 2 and 4). *Note: If the real dataset is missing, it auto-generates a statistically equivalent synthetic version.*
- `run_s3_neural.py`: **Neural Attack/Defense Evaluation**. Trains an LSTM anomaly detector, mounts adversarial PGD attacks against it, and trains a GAN evasion generator. Produces Tables 6 (detection comparison), 7 (adversarial LSTM), and 8 (GAN vs PGD).
- `generate_figures.py`: Creates the beautiful, publication-ready PDF plots (including S3 neural figures).
- `collect_results.py`: Aggregates the JSON outputs into formatted `.tex` tables.

### 3. Red-Team Emulation (`services/`)
If you want to run the system "live" as a real industrial network, we provide a Docker Compose stack.
- `plc/`: Modbus-based Programmable Logic Controller running IEC-61131-3 equivalent logic.
- `digital_twin/`: Live Python service that subscribes to network traffic, runs the EKF+IWD, and raises alarms.
- `historian/`: Logs network events.
- `docker-compose.yml`: Spins up the entire architecture. Run with `make up`.

### 4. Tests (`tests/`)
Comprehensive unit tests verifying the mathematical correctness of the detectors and attackers.
Run them using:
```bash
make test
```

---

## 📊 Understanding the Output

When you run the experiments, here is what the framework produces:

### Tables (`results/tables/`)
* **Table 1 (Evasion Rate)**: Proves that for physically meaningful faults (e.g., $2.0\sigma$ or higher), the attacker's evasion rate drops to exactly 0.0% due to the Stein divergence trap.
* **Table 2 & 4 (SWaT Validation)**: Shows the defense scales from our 6-sensor Two-Tank model to the 51-sensor SWaT industrial dataset.
* **Table 3 (Budget Sweep)**: Shows the average Sensor Deception Score (SDS) across different attacker constraints.
* **Table 5 (Ablation)**: Proves that CUSUM alone is highly vulnerable (high False Negative Rate under attack), while the combined CUSUM + ISWT pipeline perfectly secures the system.

### Figures (`results/figures/`)
* **`sds_convergence.pdf`**: The crown jewel of the paper. It plots the attacker's internal PyTorch optimization loss (red) against the true deception metric (blue). It visually proves that even when the attacker mathematically optimizes their attack perfectly, the physical constraints of our defense prevent them from succeeding.
* **`cusum_timeseries.pdf` & `iswt_timeseries.pdf`**: Demonstrates the raw detector statistics before, during, and after an attack.
* **`budget_sweep.pdf`**: Plots the attacker's budget $\epsilon$ against their success rate, contrasting White-box and Grey-box knowledge levels.
* **`innovation_covariance.pdf`**: A heatmap showing the exact cross-sensor correlations (the "fingerprint") that the TCA leaves behind, which our ISWT detector uses to catch them.

---

## 🛠 Advanced Usage

### SWaT Dataset Integration
To run S2 on the real SWaT dataset instead of the synthetic fallback:
1. Download the `SWaT_Dataset_Normal_v1.csv` and `SWaT_Dataset_Attack_v0.csv` from the [iTrust SUTD repository](https://itrust.sutd.edu.sg/itrust-labs_datasets/).
2. Place them in the `swat/data/` directory.
3. Run `make run-s2`.

### Live Docker Emulation
To interact with the live Red-Team network:
```bash
# Start the entire industrial network
make up

# (Wait for services to initialize)

# Stop the network
make down
```
