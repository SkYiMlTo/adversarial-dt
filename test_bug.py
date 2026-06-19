import numpy as np
from core.config import SystemConfig, ExperimentConfig
from core.calibration import full_calibration
from core.iswt import ISWTDetector

sys_cfg = SystemConfig()
calib = full_calibration(sys_cfg, 1000, seed=42)
print("Calibration Passed:", calib['whiteness_validation']['passed'])
iswt_cfg = calib['iswt_config']
print("Empirical Critical:", iswt_cfg.empirical_critical)

# Now test it on clean data
from core.process_model import TwoTankProcess
from core.ekf import ExtendedKalmanFilter

process = TwoTankProcess(sys_cfg)
sim = process.simulate(1000, fault_config=None, seed=43)
ekf = ExtendedKalmanFilter(sys_cfg, calib['ekf_config'])
ekf.set_noise_covariances(calib['Q'], calib['R'])
res = ekf.run_batch(sim['y_faulted'], sim['u'])

iswt = ISWTDetector(6, iswt_cfg)
iswt_res = iswt.run_batch(res['std_innovation'])
fpr = np.mean(iswt_res['alarm'][iswt_cfg.W:])
print("FPR:", fpr)
print("Mean test_stat:", np.mean(iswt_res['test_stat'][iswt_cfg.W:]))
