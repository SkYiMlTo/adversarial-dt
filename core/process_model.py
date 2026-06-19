"""
Two-tank water distribution process model.

Implements the physical process described in Sec. 4.5 of the paper:
a two-tank water distribution loop with N=6 sensors, governed by
mass-balance equations and a quadratic pump head-flow curve.

State vector:  x = [L1, L2, P_in, P_out, Q12, Q_pump]^T ∈ R^6
Control input: u = [u_pump, u_valve]

The ODE is integrated at 100 ms by a 4th-order Runge-Kutta solver.
Sensor readings are sampled at 1 Hz through the measurement model
h(x) = x (direct sensor read) plus Gaussian noise.

The Jacobian ∂f/∂x is computed analytically for the EKF.

Physical parameters justification:
    - Tank cross-sections (1.0, 0.8 m²): typical small-scale CPS testbed
    - Pump curve H = H0 - a·Q²: standard centrifugal pump model
    - Orifice flow Q = Cv·sign(ΔL)·√|ΔL|: Torricelli's law
    - Fast-state relaxation (τ = 0.5s): models sensor/actuator dynamics
"""

import numpy as np
from typing import Tuple, Optional
from .config import SystemConfig


# Small constant to avoid division-by-zero in sqrt
_EPS = 1e-8


class TwoTankProcess:
    """Continuous-time two-tank water distribution process.

    Provides:
        - ODE right-hand side f(x, u)
        - Analytical Jacobian ∂f/∂x
        - RK4 integration
        - Measurement model h(x) = x with additive noise
        - PLC control logic
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.cfg = config or SystemConfig()
        self._rng = np.random.default_rng(seed=42)

    def set_seed(self, seed: int):
        """Reset the random number generator."""
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time state derivative dx/dt = f(x, u).

        Args:
            x: State vector [L1, L2, P_in, P_out, Q12, Q_pump].
            u: Control input [u_pump, u_valve].

        Returns:
            dx: State derivative vector.
        """
        c = self.cfg
        L1, L2, P_in, P_out, Q12, Q_pump = x
        u_pump, u_valve = u

        # Ensure physical bounds (soft clamping for numerical stability)
        L1 = max(L1, 0.0)
        L2 = max(L2, 0.0)
        Q_pump = max(Q_pump, 0.0)

        # Outflow from tank 2 through valve
        Q_out = c.Cv_out * u_valve * np.sqrt(L2 + _EPS)

        # Steady-state targets for fast states
        # Hydrostatic pressure at pump inlet
        P_in_ss = L1  # In meters of water head: P/(ρg) = L1

        # Pump head-flow curve: H = H0 - a_pump · Q²
        H_pump = c.H0 - c.a_pump * Q_pump ** 2
        H_pump = max(H_pump, 0.0)  # Pump cannot produce negative head
        P_out_ss = P_in + H_pump

        # Inter-tank flow (Torricelli / orifice equation)
        dL = L1 - L2
        Q12_ss = c.Cv12 * np.sign(dL) * np.sqrt(np.abs(dL) + _EPS)

        # Pump flow target
        Q_pump_ss = u_pump * c.Q_nom

        # State derivatives
        dL1 = (Q_pump - Q12) / c.A1
        dL2 = (Q12 - Q_out) / c.A2
        dP_in = (P_in_ss - P_in) / c.tau_p
        dP_out = (P_out_ss - P_out) / c.tau_p
        dQ12 = (Q12_ss - Q12) / c.tau_q
        dQ_pump = (Q_pump_ss - Q_pump) / c.tau_q

        return np.array([dL1, dL2, dP_in, dP_out, dQ12, dQ_pump])

    # ------------------------------------------------------------------
    # Analytical Jacobian
    # ------------------------------------------------------------------

    def jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Analytical Jacobian F = ∂f/∂x evaluated at (x, u).

        Returns:
            F: 6×6 Jacobian matrix.
        """
        c = self.cfg
        L1, L2, P_in, P_out, Q12, Q_pump = x
        u_pump, u_valve = u

        L1 = max(L1, _EPS)
        L2 = max(L2, _EPS)
        Q_pump = max(Q_pump, 0.0)

        F = np.zeros((6, 6))

        # ∂(dL1/dt) / ∂x
        F[0, 4] = -1.0 / c.A1          # ∂/∂Q12
        F[0, 5] = 1.0 / c.A1           # ∂/∂Q_pump

        # ∂(dL2/dt) / ∂x
        # Q_out = Cv_out · u_valve · sqrt(L2 + ε)
        F[1, 1] = -c.Cv_out * u_valve / (2.0 * c.A2 * np.sqrt(L2 + _EPS))
        F[1, 4] = 1.0 / c.A2           # ∂/∂Q12

        # ∂(dP_in/dt) / ∂x
        # P_in_ss = L1, so ∂P_in_ss/∂L1 = 1
        F[2, 0] = 1.0 / c.tau_p        # ∂/∂L1
        F[2, 2] = -1.0 / c.tau_p       # ∂/∂P_in

        # ∂(dP_out/dt) / ∂x
        # P_out_ss = P_in + H0 - a_pump·Q_pump²
        F[3, 2] = 1.0 / c.tau_p        # ∂/∂P_in
        F[3, 3] = -1.0 / c.tau_p       # ∂/∂P_out
        # ∂/∂Q_pump: ∂(H_pump)/∂Q_pump = -2·a_pump·Q_pump
        F[3, 5] = -2.0 * c.a_pump * Q_pump / c.tau_p

        # ∂(dQ12/dt) / ∂x
        # Q12_ss = Cv12 · sign(ΔL) · sqrt(|ΔL| + ε), ΔL = L1 - L2
        # ∂Q12_ss/∂L1 = Cv12 / (2·sqrt(|ΔL| + ε))
        # ∂Q12_ss/∂L2 = -Cv12 / (2·sqrt(|ΔL| + ε))
        dL = L1 - L2
        inv_sqrt_dL = 1.0 / (2.0 * np.sqrt(np.abs(dL) + _EPS))
        F[4, 0] = c.Cv12 * inv_sqrt_dL / c.tau_q     # ∂/∂L1
        F[4, 1] = -c.Cv12 * inv_sqrt_dL / c.tau_q    # ∂/∂L2
        F[4, 4] = -1.0 / c.tau_q                      # ∂/∂Q12

        # ∂(dQ_pump/dt) / ∂x
        F[5, 5] = -1.0 / c.tau_q       # ∂/∂Q_pump

        return F

    # ------------------------------------------------------------------
    # Discrete-time state transition Jacobian for EKF
    # ------------------------------------------------------------------

    def discrete_jacobian(self, x: np.ndarray, u: np.ndarray,
                          dt: Optional[float] = None) -> np.ndarray:
        """Discrete-time state transition Jacobian Φ ≈ I + F·dt.

        Used by the EKF for covariance prediction:
            P(t+1|t) = Φ · P(t) · Φ^T + Q

        For better accuracy, uses second-order approximation:
            Φ ≈ I + F·dt + (F·dt)²/2
        """
        dt = dt or self.cfg.dt_sample
        F = self.jacobian(x, u)
        Fdt = F * dt
        Phi = np.eye(6) + Fdt + 0.5 * Fdt @ Fdt
        return Phi

    # ------------------------------------------------------------------
    # RK4 integrator
    # ------------------------------------------------------------------

    def rk4_step(self, x: np.ndarray, u: np.ndarray,
                 dt: Optional[float] = None) -> np.ndarray:
        """Single 4th-order Runge-Kutta step.

        Args:
            x: Current state.
            u: Control input (held constant during step).
            dt: Integration step size.

        Returns:
            x_next: State at t + dt.
        """
        dt = dt or self.cfg.dt_sim
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce physical constraints
        x_next[0] = np.clip(x_next[0], 0.0, self.cfg.L1_max)  # L1
        x_next[1] = np.clip(x_next[1], 0.0, self.cfg.L2_max)  # L2
        x_next[5] = max(x_next[5], 0.0)                        # Q_pump ≥ 0

        return x_next

    def integrate(self, x: np.ndarray, u: np.ndarray,
                  n_steps: Optional[int] = None) -> np.ndarray:
        """Integrate over one sample period (multiple RK4 steps).

        Runs sim_steps_per_sample RK4 steps at dt_sim to advance
        from t to t + dt_sample.

        Returns:
            x_next: State at t + dt_sample.
        """
        n_steps = n_steps or self.cfg.sim_steps_per_sample
        for _ in range(n_steps):
            x = self.rk4_step(x, u)
        return x

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------

    def measure(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Measurement model h(x) = x + η, η ~ N(0, R).

        The measurement function is the identity: each sensor directly
        reads the corresponding state variable.

        Args:
            x: True state vector.
            add_noise: Whether to add measurement noise.

        Returns:
            y: Measurement vector.
        """
        y = x.copy()
        if add_noise:
            y += self._rng.normal(0.0, self.cfg.sigma)
        return y

    @staticmethod
    def measurement_jacobian(n: int = 6) -> np.ndarray:
        """Measurement Jacobian H = ∂h/∂x = I_N.

        Since h(x) = x (direct measurement), H is the identity matrix.
        """
        return np.eye(n)

    # ------------------------------------------------------------------
    # PLC control logic
    # ------------------------------------------------------------------

    def control_logic(self, y: np.ndarray,
                      u_prev: np.ndarray) -> np.ndarray:
        """Simple level-based PLC control logic (IEC 61131-3 equivalent).

        Control rules:
            - Pump ON  when L1 < L1_low  (hysteresis control)
            - Pump OFF when L1 > L1_high
            - Valve position: constant at valve_open

        Args:
            y: Current sensor readings (measurement vector).
            u_prev: Previous control input [u_pump, u_valve].

        Returns:
            u: Updated control input.
        """
        c = self.cfg
        L1_measured = y[0]
        u_pump = u_prev[0]

        # Hysteresis-based pump control
        if L1_measured < c.L1_low:
            u_pump = 1.0
        elif L1_measured > c.L1_high:
            u_pump = 0.0
        # else: maintain previous state (hysteresis)

        u_valve = c.valve_open
        return np.array([u_pump, u_valve])

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self, n_steps: int,
                 x0: Optional[np.ndarray] = None,
                 u0: Optional[np.ndarray] = None,
                 fault_config: Optional[dict] = None,
                 seed: Optional[int] = None) -> dict:
        """Run a full simulation of the two-tank process.

        Args:
            n_steps: Number of sample-period steps to simulate.
            x0: Initial state (defaults to steady-state config).
            u0: Initial control input (defaults to [1, valve_open]).
            fault_config: Optional dict with keys:
                - 'sensor_idx': list of sensor indices to fault
                - 'fault_start': timestep to start fault
                - 'fault_magnitude': magnitude in absolute units
                  (or list of magnitudes per sensor)
            seed: Random seed for measurement noise.

        Returns:
            Dictionary with arrays:
                - 'x_true': (n_steps, 6) true states
                - 'y_clean': (n_steps, 6) noise-free measurements
                - 'y_noisy': (n_steps, 6) noisy measurements
                - 'y_faulted': (n_steps, 6) measurements with fault applied
                - 'u': (n_steps, 2) control inputs
                - 'faults': (n_steps, 6) fault vectors
        """
        if seed is not None:
            self.set_seed(seed)

        c = self.cfg
        x = x0.copy() if x0 is not None else c.x0.copy()
        u = u0.copy() if u0 is not None else np.array([1.0, c.valve_open])

        # Pre-allocate arrays
        X = np.zeros((n_steps, c.n_states))
        Y_clean = np.zeros((n_steps, c.n_sensors))
        Y_noisy = np.zeros((n_steps, c.n_sensors))
        Y_faulted = np.zeros((n_steps, c.n_sensors))
        U = np.zeros((n_steps, 2))
        Faults = np.zeros((n_steps, c.n_sensors))

        for t in range(n_steps):
            # Record state
            X[t] = x

            # Measure
            y_clean = self.measure(x, add_noise=False)
            y_noisy = self.measure(x, add_noise=True)
            Y_clean[t] = y_clean
            Y_noisy[t] = y_noisy

            # Apply physical fault
            y_faulted = y_noisy.copy()
            if fault_config is not None and t >= fault_config.get('fault_start', n_steps):
                for idx in fault_config['sensor_idx']:
                    mag = fault_config['fault_magnitude']
                    if isinstance(mag, (list, np.ndarray)):
                        delta = mag[idx]
                    else:
                        # Scale by σ_i for the specific sensor
                        delta = mag * c.sigma[idx]
                    y_faulted[idx] += delta
                    Faults[t, idx] = delta
            Y_faulted[t] = y_faulted

            # Control logic (uses potentially faulted readings,
            # since PLC sees what the sensors report)
            u = self.control_logic(y_faulted, u)
            U[t] = u

            # Integrate state
            x = self.integrate(x, u)

        return {
            'x_true': X,
            'y_clean': Y_clean,
            'y_noisy': Y_noisy,
            'y_faulted': Y_faulted,
            'u': U,
            'faults': Faults,
        }
