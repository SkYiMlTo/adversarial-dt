"""
Configuration for the two-tank CPS testbed and experiment parameters.

All physical parameters are drawn from standard hydraulic references
for small-scale water distribution systems used in CPS security research
(cf. MiniCPS, SWaT literature).

Units: SI throughout (meters, seconds, kg, m³/s, meters of water head).
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO = 1000.0   # Water density [kg/m³]
G = 9.81       # Gravitational acceleration [m/s²]


@dataclass
class SystemConfig:
    """Two-tank water distribution system parameters.

    State vector: x = [L1, L2, P_in, P_out, Q12, Q_pump]^T ∈ R^6
        L1      — Main tank water level [m]
        L2      — Secondary tank water level [m]
        P_in    — Pump inlet pressure [m of water head]
        P_out   — Pump outlet pressure [m of water head]
        Q12     — Inter-tank flow rate [m³/s]
        Q_pump  — Pump discharge flow rate [m³/s]

    Control inputs: u = [u_pump, u_valve]
        u_pump  ∈ {0, 1}   — Pump enable command
        u_valve ∈ [0, 1]   — Valve position (0=closed, 1=fully open)
    """

    # --- Tank geometry ---
    A1: float = 1.0          # Tank 1 cross-sectional area [m²]
    A2: float = 0.8          # Tank 2 cross-sectional area [m²]
    L1_max: float = 5.0      # Tank 1 maximum level [m]
    L2_max: float = 4.0      # Tank 2 maximum level [m]

    # --- Initial conditions ---
    L1_init: float = 2.5     # Initial level tank 1 [m]
    L2_init: float = 2.0     # Initial level tank 2 [m]

    # --- Pump parameters (quadratic head-flow curve) ---
    # H_pump(Q) = H0 - a_pump · Q²
    H0: float = 20.0         # Shutoff head [m]
    Q_rated: float = 0.01    # Rated flow at zero head [m³/s]
    Q_nom: float = 0.008     # Nominal operating flow [m³/s]

    # --- Pipe / orifice parameters ---
    # Q12 = Cv12 · sign(L1-L2) · sqrt(|L1-L2|)
    Cv12: float = 0.005      # Inter-tank orifice coefficient [m^(5/2)/s]
    # Q_out = Cv_out · u_valve · sqrt(L2)
    Cv_out: float = 0.003    # Outflow orifice coefficient [m^(5/2)/s]

    # --- Sensor / actuator dynamics (fast-state relaxation) ---
    tau_p: float = 0.5       # Pressure sensor time constant [s]
    tau_q: float = 0.5       # Flow sensor time constant [s]

    # --- PLC control thresholds ---
    L1_low: float = 1.0      # Pump ON when L1 < L1_low [m]
    L1_high: float = 4.0     # Pump OFF when L1 > L1_high [m]
    valve_open: float = 0.6  # Default valve position

    # --- Simulation timing ---
    dt_sim: float = 0.1      # ODE integration step (RK4) [s]
    dt_sample: float = 1.0   # Modbus / EKF sample period [s]

    # --- Measurement noise ---
    # σ_η derived from Modbus 16-bit integer quantization:
    # σ_i ≈ 0.5% of full-scale range per sensor.
    # This matches the paper's Sec 4.5: "measurement noise covariance R
    # is diagonal with entries set from the Modbus register quantization step."
    sigma: np.ndarray = field(default_factory=lambda: np.array([
        0.025,    # L1:     range [0, 5] m      → 0.5% × 5 = 0.025
        0.020,    # L2:     range [0, 4] m      → 0.5% × 4 = 0.020
        0.025,    # P_in:   range [0, 5] m_head → 0.5% × 5 = 0.025
        0.150,    # P_out:  range [0, 30] m_head → 0.5% × 30 = 0.15
        0.0002,   # Q12:    range [-0.02, 0.02] m³/s → 0.5% × 0.04
        0.0001,   # Q_pump: range [0, 0.02] m³/s → 0.5% × 0.02
    ]))

    # --- Derived quantities ---
    @property
    def n_sensors(self) -> int:
        return 6

    @property
    def n_states(self) -> int:
        return 6

    @property
    def a_pump(self) -> float:
        """Quadratic pump curve coefficient: H = H0 - a_pump · Q²."""
        return self.H0 / (self.Q_rated ** 2)

    @property
    def R(self) -> np.ndarray:
        """Measurement noise covariance matrix (diagonal)."""
        return np.diag(self.sigma ** 2)

    @property
    def x0(self) -> np.ndarray:
        """Initial state vector consistent with steady-state physics."""
        L1 = self.L1_init
        L2 = self.L2_init
        P_in = L1  # hydrostatic: P_in = ρgL1 / (ρg) = L1 in m_head
        Q_pump = self.Q_nom
        H_pump = self.H0 - self.a_pump * Q_pump ** 2
        P_out = P_in + H_pump
        dL = L1 - L2
        Q12 = self.Cv12 * np.sign(dL) * np.sqrt(np.abs(dL))
        return np.array([L1, L2, P_in, P_out, Q12, Q_pump])

    @property
    def sim_steps_per_sample(self) -> int:
        return int(self.dt_sample / self.dt_sim)

    # --- Sensor metadata (for Modbus register scaling) ---
    @property
    def sensor_names(self) -> list:
        return ["L1", "L2", "P_in", "P_out", "Q12", "Q_pump"]

    @property
    def sensor_ranges(self) -> list:
        """(min, max) for each sensor, used for Modbus 16-bit scaling."""
        return [
            (0.0, 5.0),      # L1 [m]
            (0.0, 4.0),      # L2 [m]
            (0.0, 5.0),      # P_in [m_head]
            (0.0, 30.0),     # P_out [m_head]
            (-0.02, 0.02),   # Q12 [m³/s]
            (0.0, 0.02),     # Q_pump [m³/s]
        ]


@dataclass
class EKFConfig:
    """Extended Kalman Filter parameters.

    The EKF runs the two-tank process model as its internal prediction,
    with the measurement model h(x) = x (direct sensor read, identity).
    """

    # --- Process noise covariance ---
    # Q is estimated from clean operational data by ML estimation
    # (see calibration.py).  Initial guess based on model uncertainty.
    Q_diag: np.ndarray = field(default_factory=lambda: np.array([
        1e-4,    # L1 model uncertainty
        1e-4,    # L2 model uncertainty
        1e-3,    # P_in model uncertainty
        1e-3,    # P_out model uncertainty
        1e-6,    # Q12 model uncertainty
        1e-6,    # Q_pump model uncertainty
    ]))

    # --- Initial error covariance ---
    P0_diag: np.ndarray = field(default_factory=lambda: np.array([
        0.01, 0.01, 0.01, 0.1, 1e-5, 1e-5
    ]))

    @property
    def Q(self) -> np.ndarray:
        return np.diag(self.Q_diag)

    @property
    def P0(self) -> np.ndarray:
        return np.diag(self.P0_diag)


@dataclass
class CUSUMConfig:
    """Per-sensor CUSUM detector parameters.

    The CUSUM is sequentially optimal for detecting persistent mean shifts
    in the innovation sequence (Lorden minimax theorem).
    """
    k: float = 0.5   # Reference value (detects shifts ≥ 1σ)
    h: float = 5.0   # Alarm threshold (ARL₀ = e^{2kh} ≈ 148 steps)

    @property
    def arl0(self) -> float:
        """Average run length under H0 (Wald-Lorden bound)."""
        return np.exp(2 * self.k * self.h)


@dataclass
class ISWTConfig:
    """Innovation Spatial Whiteness Test parameters.

    The ISWT uses the Stein matrix divergence with a chi-squared null
    distribution (Bartlett's theorem).
    """
    W: int = 200         # Sliding window length [timesteps]
    alpha: float = 0.05  # Significance level
    empirical_critical: float = None  # Empirically calibrated threshold

    def critical_value(self, n_sensors: int) -> float:
        """Critical value for the ISWT alarm.

        Uses empirical calibration if available, else theoretical chi-squared.
        """
        if self.empirical_critical is not None:
            return self.empirical_critical
        from scipy.stats import chi2
        dof = n_sensors * (n_sensors + 1) // 2
        return chi2.ppf(1 - self.alpha, dof)


@dataclass
class TCAConfig:
    """Targeted Consistency Attack parameters."""
    K: int = 100                    # Number of PGD iterations
    eta: float = 0.003              # Step size (Armijo line-search init)
    fd_step: float = 5e-4           # Finite difference step (grey-box)
    armijo_c: float = 1e-4          # Armijo sufficient decrease constant
    armijo_rho: float = 0.5         # Armijo backtracking factor

    # Budget sweep: ε / σ_η
    epsilon_ratios: list = field(default_factory=lambda: [
        0.25, 0.50, 0.75, 1.00, 1.50
    ])


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    system: SystemConfig = field(default_factory=SystemConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)
    iswt: ISWTConfig = field(default_factory=ISWTConfig)
    tca: TCAConfig = field(default_factory=TCAConfig)

    # --- S1 red-team protocol ---
    s1_sessions_per_config: int = 30       # Sessions per (regime, fault_mag)
    s1_session_duration_steps: int = 600   # 10 min at 1 Hz
    s1_evasion_window: int = 60            # 60 consecutive alarm-free steps
    s1_calibration_steps: int = 1800       # 30 min at 1 Hz
    s1_fault_magnitudes: list = field(default_factory=lambda: [1.0, 2.0, 4.0])
    # Fault magnitudes in multiples of σ_η

    # --- S2 SWaT ---
    s2_train_steps: int = 72000
    s2_n_attack_sensors: int = 4  # |A| = 4 sensors per stage

    # --- Random seed ---
    seed: int = 42

    # --- Parallelization ---
    n_workers: int = 10  # Parallel sessions
