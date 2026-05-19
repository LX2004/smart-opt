from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NumberOrArray = Union[float, int, Iterable[float], np.ndarray]


@dataclass
class HOBParams:
    """
    Reduced HOB-SCP model parameters.

    HOB:
        Hydrogen-oxidizing bacteria.

    SCP:
        Single-cell protein.

    Units:
        mu: h^-1
        q: mmol / gDW / h
        substrate_per_gdw: mmol substrate / gDW biomass
    """

    # Growth and composition
    mu_max: float = 0.12
    f_protein: float = 0.65

    # Approximate biomass stoichiometry
    h2_per_gdw: float = 318.0
    o2_per_gdw: float = 118.0
    co2_per_gdw: float = 41.0

    # Non-growth maintenance demand
    m_h2: float = 2.0
    m_o2: float = 0.75

    # Biological capacity constraints
    q_h2_hydrogenase_cap: float = 45.0
    q_o2_etc_cap: float = 18.0
    q_co2_cbb_cap: float = 6.0

    # Biosynthetic capacity cap
    mu_proteome_cap: float = 0.12

    # Starvation decay
    k_starvation: float = 0.03


@dataclass
class HOBLoadConfig:
    """
    Dispatch and replacement settings.

    maintenance_fraction_cutoff:
        If H2_maintenance_fraction >= this value, and the current batch has
        already operated longer than min_runtime_before_replacement_h, then
        the current bacterial load is considered inefficient and is replaced.

        Default 0.5 means:
        when more than half of the consumed H2 is used for maintenance instead
        of new SCP growth, cut off the current load and restart a new batch.

    min_runtime_before_replacement_h:
        Minimum operating time before automatic replacement is allowed.

    auto_replace:
        If True, replacement is triggered automatically by the maintenance
        fraction cutoff.

    restart_immediately_after_replacement:
        If True, start a new initialized batch immediately after replacement.

    reset_on_shutdown:
        If True, external shutdown cuts off the current batch.
        If False, external shutdown pauses gas consumption but keeps biomass.
    """

    maintenance_fraction_cutoff: float = 0.5
    min_runtime_before_replacement_h: float = 24.0
    auto_replace: bool = True
    restart_immediately_after_replacement: bool = True
    reset_on_shutdown: bool = True


class HOBHydrogenLoad:
    """
    Dispatchable HOB hydrogen load.

    Power-system interpretation:
        This object behaves like a dynamic hydrogen load.

        At each time step, the scheduler sends:
            - load_on
            - F_H2_mol_h
            - F_O2_mol_h
            - F_CO2_mol_h

        The model returns:
            - H2_load_mol_h

        H2_load_mol_h is the actual hydrogen consumption rate and can be used
        directly in a hydrogen storage balance.

    Example:
        h2_tank_next = (
            h2_tank_now
            + h2_from_electrolyzer_mol_h * dt_h
            - h2_to_fuel_cell_mol_h * dt_h
            - H2_load_mol_h * dt_h
        )
    """

    def __init__(
        self,
        X0_gdw: float,
        params: Optional[HOBParams] = None,
        load_config: Optional[HOBLoadConfig] = None,
        start_active: bool = True,
    ) -> None:
        if X0_gdw <= 0:
            raise ValueError("X0_gdw must be > 0")

        self.params = params or HOBParams()
        self.config = load_config or HOBLoadConfig()

        if not (0.0 <= self.config.maintenance_fraction_cutoff <= 1.0):
            raise ValueError("maintenance_fraction_cutoff must be between 0 and 1")

        if self.config.min_runtime_before_replacement_h < 0:
            raise ValueError("min_runtime_before_replacement_h must be non-negative")

        self.X0_gdw = float(X0_gdw)
        self.initial_scp_g = self.params.f_protein * self.X0_gdw

        self.cycle_id = 0
        self.cumulative_harvested_scp_g = 0.0
        self.records = []

        self.active = False
        self.X_gdw = 0.0
        self.SCP_g_protein = 0.0
        self.age_h = 0.0

        self.last_maintenance_fraction = 0.0
        self.last_growth_fraction = 0.0

        if start_active:
            self.start_new_batch()

    @staticmethod
    def _profile(x: NumberOrArray, n: int) -> np.ndarray:
        """
        Convert scalar or array-like input to a length-n numpy array.
        """
        if np.isscalar(x):
            return np.full(n, float(x), dtype=float)

        arr = np.asarray(x, dtype=float)

        if len(arr) != n:
            raise ValueError(f"Input profile length must be {n}, got {len(arr)}")

        return arr

    def start_new_batch(self) -> None:
        """
        Initialize a fresh biological hydrogen load.
        """
        self.active = True
        self.X_gdw = self.X0_gdw
        self.SCP_g_protein = self.initial_scp_g
        self.age_h = 0.0
        self.last_maintenance_fraction = 0.0
        self.last_growth_fraction = 0.0

    def shutdown(self, harvest: bool = True) -> float:
        """
        Shut down the current biological load.

        Parameters
        ----------
        harvest:
            If True, harvest the SCP increment above the initial SCP.

        Returns
        -------
        harvested_scp_g:
            Harvested SCP increment, g protein.
        """
        harvested_scp_g = 0.0

        if self.active and harvest:
            harvested_scp_g = max(0.0, self.SCP_g_protein - self.initial_scp_g)
            self.cumulative_harvested_scp_g += harvested_scp_g

        self.active = False
        self.X_gdw = 0.0
        self.SCP_g_protein = 0.0
        self.age_h = 0.0
        self.last_maintenance_fraction = 0.0
        self.last_growth_fraction = 0.0

        return harvested_scp_g

    def replace_batch(self) -> float:
        """
        Cut off the current batch and optionally start a fresh batch.

        Returns
        -------
        harvested_scp_g:
            Harvested SCP increment, g protein.
        """
        harvested_scp_g = 0.0

        if self.active:
            harvested_scp_g = max(0.0, self.SCP_g_protein - self.initial_scp_g)
            self.cumulative_harvested_scp_g += harvested_scp_g

        old_cycle_id = self.cycle_id
        self.cycle_id += 1

        if self.config.restart_immediately_after_replacement:
            self.start_new_batch()
        else:
            self.active = False
            self.X_gdw = 0.0
            self.SCP_g_protein = 0.0
            self.age_h = 0.0
            self.last_maintenance_fraction = 0.0
            self.last_growth_fraction = 0.0

        return harvested_scp_g

    def state(self) -> Dict[str, float]:
        """
        Return current state for optimization or reinforcement learning.
        """
        return {
            "active": float(self.active),
            "cycle_id": float(self.cycle_id),
            "X_gDW": self.X_gdw,
            "SCP_g_protein": self.SCP_g_protein,
            "age_h": self.age_h,
            "last_H2_maintenance_fraction": self.last_maintenance_fraction,
            "last_H2_growth_fraction": self.last_growth_fraction,
            "cumulative_harvested_SCP_g_protein": self.cumulative_harvested_scp_g,
        }

    def _off_record(
        self,
        time_h: float,
        dt_h: float,
        F_H2_mol_h: float,
        F_O2_mol_h: float,
        F_CO2_mol_h: float,
        load_on: bool,
        shutdown_event: bool = False,
        restart_event: bool = False,
        harvested_scp_g: float = 0.0,
        reason: str = "off",
    ) -> Dict[str, object]:
        """
        Create a standard zero-consumption record when the load is off.
        """
        return {
            "time_h": time_h,
            "dt_h": dt_h,
            "cycle_id": self.cycle_id,
            "load_on_command": bool(load_on),
            "load_active": bool(self.active),
            "shutdown_event": bool(shutdown_event),
            "restart_event": bool(restart_event),
            "replacement_event": False,
            "replacement_reason": reason,
            "age_h": self.age_h,

            "X_gDW": self.X_gdw,
            "SCP_g_protein": self.SCP_g_protein,
            "X_gDW_after_event": self.X_gdw,
            "SCP_g_protein_after_event": self.SCP_g_protein,

            "dX_gDW": 0.0,
            "dSCP_g_protein": 0.0,
            "mu_h-1": 0.0,
            "limiting_factor": reason,
            "starvation_level": 0.0,

            "H2_input_mol_h": float(F_H2_mol_h),
            "O2_input_mol_h": float(F_O2_mol_h),
            "CO2_input_mol_h": float(F_CO2_mol_h),

            "H2_in_mol": float(F_H2_mol_h) * dt_h,
            "O2_in_mol": float(F_O2_mol_h) * dt_h,
            "CO2_in_mol": float(F_CO2_mol_h) * dt_h,

            "q_H2_supply_mmol_gDW_h": 0.0,
            "q_O2_supply_mmol_gDW_h": 0.0,
            "q_CO2_supply_mmol_gDW_h": 0.0,

            "q_H2_available_mmol_gDW_h": 0.0,
            "q_O2_available_mmol_gDW_h": 0.0,
            "q_CO2_available_mmol_gDW_h": 0.0,

            "H2_hydrogenase_cap_active": False,
            "O2_ETC_cap_active": False,
            "CO2_CBB_cap_active": False,

            "H2_load_mol_h": 0.0,
            "H2_uptake_mol": 0.0,
            "O2_uptake_mol": 0.0,
            "CO2_uptake_mol": 0.0,

            "H2_maintenance_required_mol": 0.0,
            "O2_maintenance_required_mol": 0.0,

            "H2_maintenance_mol": 0.0,
            "H2_growth_mol": 0.0,
            "H2_maintenance_fraction": 0.0,
            "H2_growth_fraction": 0.0,

            "O2_maintenance_mol": 0.0,
            "O2_growth_mol": 0.0,
            "O2_maintenance_fraction": 0.0,
            "O2_growth_fraction": 0.0,

            "CO2_growth_mol": 0.0,

            "H2_unused_mol": float(F_H2_mol_h) * dt_h,
            "O2_unused_mol": float(F_O2_mol_h) * dt_h,
            "CO2_unused_mol": float(F_CO2_mol_h) * dt_h,

            "SCP_per_H2_uptake_g_per_mol": 0.0,

            "harvested_SCP_increment_g_protein": harvested_scp_g,
            "cumulative_harvested_SCP_g_protein": self.cumulative_harvested_scp_g,
        }

    def step(
        self,
        F_H2_mol_h: float,
        F_O2_mol_h: float,
        F_CO2_mol_h: float,
        dt_h: float = 1.0,
        load_on: bool = True,
        force_replace: bool = False,
        time_h: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Run one dispatch step.

        Parameters
        ----------
        F_H2_mol_h:
            Available H2 supply rate to this load, mol/h.

        F_O2_mol_h:
            Available O2 supply rate, mol/h.

        F_CO2_mol_h:
            Available CO2 supply rate, mol/h.

        dt_h:
            Time step, h.

        load_on:
            External dispatch command.
            True means the hydrogen load is allowed to operate.
            False means the hydrogen load is off.

        force_replace:
            If True, force replacement at this step.

        time_h:
            Optional timestamp for logging.

        Returns
        -------
        row:
            A dictionary containing the full state and flow record.
            The key scheduling output is row["H2_load_mol_h"].
        """
        if dt_h <= 0:
            raise ValueError("dt_h must be > 0")

        if F_H2_mol_h < 0 or F_O2_mol_h < 0 or F_CO2_mol_h < 0:
            raise ValueError("Gas input rates must be non-negative")

        if time_h is None:
            time_h = (len(self.records) + 1) * dt_h

        # ------------------------------------------------------------
        # External shutdown: no hydrogen consumption.
        # ------------------------------------------------------------
        if not load_on:
            harvested_scp_g = 0.0
            shutdown_event = False

            if self.active and self.config.reset_on_shutdown:
                harvested_scp_g = self.shutdown(harvest=True)
                shutdown_event = True

            row = self._off_record(
                time_h=time_h,
                dt_h=dt_h,
                F_H2_mol_h=F_H2_mol_h,
                F_O2_mol_h=F_O2_mol_h,
                F_CO2_mol_h=F_CO2_mol_h,
                load_on=False,
                shutdown_event=shutdown_event,
                harvested_scp_g=harvested_scp_g,
                reason="shutdown",
            )

            self.records.append(row)
            return row

        # ------------------------------------------------------------
        # If load is switched on while inactive, initialize new batch.
        # ------------------------------------------------------------
        restart_event = False

        if load_on and not self.active:
            self.start_new_batch()
            restart_event = True

        p = self.params

        X_before = float(self.X_gdw)
        SCP_before = float(self.SCP_g_protein)
        age_before = float(self.age_h)

        # ------------------------------------------------------------
        # 1. Gas input per unit biomass
        # mol/h -> mmol/gDW/h
        # ------------------------------------------------------------
        q_h2_supply = 1000.0 * F_H2_mol_h / max(X_before, 1e-12)
        q_o2_supply = 1000.0 * F_O2_mol_h / max(X_before, 1e-12)
        q_co2_supply = 1000.0 * F_CO2_mol_h / max(X_before, 1e-12)

        # ------------------------------------------------------------
        # 2. Biological capacity constraints
        # ------------------------------------------------------------
        q_h2_available = min(q_h2_supply, p.q_h2_hydrogenase_cap)
        q_o2_available = min(q_o2_supply, p.q_o2_etc_cap)
        q_co2_available = min(q_co2_supply, p.q_co2_cbb_cap)

        h2_capacity_active = q_h2_supply > p.q_h2_hydrogenase_cap
        o2_capacity_active = q_o2_supply > p.q_o2_etc_cap
        co2_capacity_active = q_co2_supply > p.q_co2_cbb_cap

        # ------------------------------------------------------------
        # 3. Maintenance demand
        # ------------------------------------------------------------
        h2_maintenance_required_mol = X_before * p.m_h2 * dt_h / 1000.0
        o2_maintenance_required_mol = X_before * p.m_o2 * dt_h / 1000.0

        phi_maint = min(
            q_h2_available / p.m_h2 if p.m_h2 > 0 else 1.0,
            q_o2_available / p.m_o2 if p.m_o2 > 0 else 1.0,
            1.0,
        )

        # ------------------------------------------------------------
        # 4. Starvation or growth
        # ------------------------------------------------------------
        if phi_maint < 1.0:
            mu = 0.0
            starvation = 1.0 - phi_maint
            decay = p.k_starvation * starvation

            dX = -decay * X_before * dt_h
            dSCP = p.f_protein * dX

            h2_maintenance_mol = h2_maintenance_required_mol * phi_maint
            o2_maintenance_mol = o2_maintenance_required_mol * phi_maint

            h2_growth_mol = 0.0
            o2_growth_mol = 0.0
            co2_growth_mol = 0.0

            h2_uptake_mol = h2_maintenance_mol
            o2_uptake_mol = o2_maintenance_mol
            co2_uptake_mol = 0.0

            limiting = "maintenance/starvation"

        else:
            starvation = 0.0

            q_h2_growth_available = max(0.0, q_h2_available - p.m_h2)
            q_o2_growth_available = max(0.0, q_o2_available - p.m_o2)

            mu_candidates = {
                "H2": q_h2_growth_available / p.h2_per_gdw,
                "O2": q_o2_growth_available / p.o2_per_gdw,
                "CO2/CBB": q_co2_available / p.co2_per_gdw,
                "mu_max": p.mu_max,
                "proteome": p.mu_proteome_cap,
            }

            limiting = min(mu_candidates, key=mu_candidates.get)
            mu = max(0.0, mu_candidates[limiting])

            dX = mu * X_before * dt_h
            dSCP = p.f_protein * dX

            h2_maintenance_mol = h2_maintenance_required_mol
            o2_maintenance_mol = o2_maintenance_required_mol

            h2_growth_mol = X_before * p.h2_per_gdw * mu * dt_h / 1000.0
            o2_growth_mol = X_before * p.o2_per_gdw * mu * dt_h / 1000.0
            co2_growth_mol = X_before * p.co2_per_gdw * mu * dt_h / 1000.0

            h2_uptake_mol = h2_maintenance_mol + h2_growth_mol
            o2_uptake_mol = o2_maintenance_mol + o2_growth_mol
            co2_uptake_mol = co2_growth_mol

        # ------------------------------------------------------------
        # 5. Numerical guard: cannot consume more than supplied
        # ------------------------------------------------------------
        h2_uptake_mol = min(h2_uptake_mol, F_H2_mol_h * dt_h)
        o2_uptake_mol = min(o2_uptake_mol, F_O2_mol_h * dt_h)
        co2_uptake_mol = min(co2_uptake_mol, F_CO2_mol_h * dt_h)

        # ------------------------------------------------------------
        # 6. Fractions
        # ------------------------------------------------------------
        h2_maintenance_fraction = h2_maintenance_mol / max(h2_uptake_mol, 1e-12)
        h2_growth_fraction = h2_growth_mol / max(h2_uptake_mol, 1e-12)

        o2_maintenance_fraction = o2_maintenance_mol / max(o2_uptake_mol, 1e-12)
        o2_growth_fraction = o2_growth_mol / max(o2_uptake_mol, 1e-12)

        h2_maintenance_fraction = float(np.clip(h2_maintenance_fraction, 0.0, 1.0))
        h2_growth_fraction = float(np.clip(h2_growth_fraction, 0.0, 1.0))

        o2_maintenance_fraction = float(np.clip(o2_maintenance_fraction, 0.0, 1.0))
        o2_growth_fraction = float(np.clip(o2_growth_fraction, 0.0, 1.0))

        # ------------------------------------------------------------
        # 7. Update biological state before replacement
        # ------------------------------------------------------------
        X_after_growth = max(0.0, X_before + dX)
        SCP_after_growth = max(0.0, SCP_before + dSCP)
        age_after_growth = age_before + dt_h

        self.X_gdw = X_after_growth
        self.SCP_g_protein = SCP_after_growth
        self.age_h = age_after_growth
        self.last_maintenance_fraction = h2_maintenance_fraction
        self.last_growth_fraction = h2_growth_fraction

        # ------------------------------------------------------------
        # 8. Replacement logic
        # ------------------------------------------------------------
        replacement_event = False
        replacement_reason = "none"
        harvested_scp_g = 0.0
        cycle_id_before_event = self.cycle_id

        inefficient_by_maintenance = (
            self.config.auto_replace
            and age_after_growth >= self.config.min_runtime_before_replacement_h
            and h2_maintenance_fraction >= self.config.maintenance_fraction_cutoff
        )

        if force_replace:
            replacement_event = True
            replacement_reason = "forced"

        elif inefficient_by_maintenance:
            replacement_event = True
            replacement_reason = "maintenance_fraction_cutoff"

        if replacement_event:
            harvested_scp_g = self.replace_batch()

        # ------------------------------------------------------------
        # 9. Record result
        # ------------------------------------------------------------
        scp_per_h2 = dSCP / max(h2_uptake_mol, 1e-12)

        row = {
            "time_h": time_h,
            "dt_h": dt_h,
            "cycle_id": cycle_id_before_event,

            "load_on_command": bool(load_on),
            "load_active": True,
            "shutdown_event": False,
            "restart_event": restart_event,
            "replacement_event": bool(replacement_event),
            "replacement_reason": replacement_reason,
            "age_h": age_after_growth,

            # Biological state before/after replacement event
            "X_gDW": X_after_growth,
            "SCP_g_protein": SCP_after_growth,
            "X_gDW_after_event": self.X_gdw,
            "SCP_g_protein_after_event": self.SCP_g_protein,

            "dX_gDW": dX,
            "dSCP_g_protein": dSCP,
            "mu_h-1": mu,
            "limiting_factor": limiting,
            "starvation_level": starvation,

            # Gas input
            "H2_input_mol_h": float(F_H2_mol_h),
            "O2_input_mol_h": float(F_O2_mol_h),
            "CO2_input_mol_h": float(F_CO2_mol_h),

            "H2_in_mol": float(F_H2_mol_h) * dt_h,
            "O2_in_mol": float(F_O2_mol_h) * dt_h,
            "CO2_in_mol": float(F_CO2_mol_h) * dt_h,

            # Specific supply
            "q_H2_supply_mmol_gDW_h": q_h2_supply,
            "q_O2_supply_mmol_gDW_h": q_o2_supply,
            "q_CO2_supply_mmol_gDW_h": q_co2_supply,

            "q_H2_available_mmol_gDW_h": q_h2_available,
            "q_O2_available_mmol_gDW_h": q_o2_available,
            "q_CO2_available_mmol_gDW_h": q_co2_available,

            "H2_hydrogenase_cap_active": h2_capacity_active,
            "O2_ETC_cap_active": o2_capacity_active,
            "CO2_CBB_cap_active": co2_capacity_active,

            # Actual uptake / dispatchable hydrogen load
            "H2_load_mol_h": h2_uptake_mol / dt_h,
            "H2_uptake_mol": h2_uptake_mol,
            "O2_uptake_mol": o2_uptake_mol,
            "CO2_uptake_mol": co2_uptake_mol,

            # Maintenance and growth split
            "H2_maintenance_required_mol": h2_maintenance_required_mol,
            "O2_maintenance_required_mol": o2_maintenance_required_mol,

            "H2_maintenance_mol": h2_maintenance_mol,
            "H2_growth_mol": h2_growth_mol,
            "H2_maintenance_fraction": h2_maintenance_fraction,
            "H2_growth_fraction": h2_growth_fraction,

            "O2_maintenance_mol": o2_maintenance_mol,
            "O2_growth_mol": o2_growth_mol,
            "O2_maintenance_fraction": o2_maintenance_fraction,
            "O2_growth_fraction": o2_growth_fraction,

            "CO2_growth_mol": co2_growth_mol,

            # Unused gas
            "H2_unused_mol": max(0.0, F_H2_mol_h * dt_h - h2_uptake_mol),
            "O2_unused_mol": max(0.0, F_O2_mol_h * dt_h - o2_uptake_mol),
            "CO2_unused_mol": max(0.0, F_CO2_mol_h * dt_h - co2_uptake_mol),

            # Efficiency and harvest
            "SCP_per_H2_uptake_g_per_mol": scp_per_h2,
            "harvested_SCP_increment_g_protein": harvested_scp_g,
            "cumulative_harvested_SCP_g_protein": self.cumulative_harvested_scp_g,
        }

        self.records.append(row)
        return row

    def simulate(
        self,
        F_H2_mol_h: NumberOrArray,
        F_O2_mol_h: NumberOrArray,
        F_CO2_mol_h: NumberOrArray,
        hours: float = 240.0,
        dt_h: float = 1.0,
        load_on_profile: Optional[NumberOrArray] = None,
        force_replace_profile: Optional[NumberOrArray] = None,
        clear_records: bool = True,
    ) -> pd.DataFrame:
        """
        Run multi-step simulation.

        Parameters
        ----------
        F_H2_mol_h, F_O2_mol_h, F_CO2_mol_h:
            Gas input rates.
            Each can be scalar or a length-n array.

        load_on_profile:
            Optional on/off profile.
            1 means load is on.
            0 means load is off.

        force_replace_profile:
            Optional replacement command profile.
            1 means force replacement at this time step.

        Returns
        -------
        pandas.DataFrame
        """
        if hours <= 0:
            raise ValueError("hours must be > 0")

        if dt_h <= 0:
            raise ValueError("dt_h must be > 0")

        if clear_records:
            self.records = []

        n = int(np.ceil(hours / dt_h))

        H2 = self._profile(F_H2_mol_h, n)
        O2 = self._profile(F_O2_mol_h, n)
        CO2 = self._profile(F_CO2_mol_h, n)

        if np.any(H2 < 0) or np.any(O2 < 0) or np.any(CO2 < 0):
            raise ValueError("Gas input rates must be non-negative")

        if load_on_profile is None:
            load_on_arr = np.ones(n, dtype=bool)
        else:
            load_on_arr = self._profile(load_on_profile, n).astype(bool)

        if force_replace_profile is None:
            force_replace_arr = np.zeros(n, dtype=bool)
        else:
            force_replace_arr = self._profile(force_replace_profile, n).astype(bool)

        rows = []

        for i in range(n):
            row = self.step(
                F_H2_mol_h=float(H2[i]),
                F_O2_mol_h=float(O2[i]),
                F_CO2_mol_h=float(CO2[i]),
                dt_h=dt_h,
                load_on=bool(load_on_arr[i]),
                force_replace=bool(force_replace_arr[i]),
                time_h=(i + 1) * dt_h,
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def summarize(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Return one-row summary statistics.
        """
        if df is None:
            df = pd.DataFrame(self.records)

        if df.empty:
            raise ValueError("No records to summarize")

        total_h2_uptake = df["H2_uptake_mol"].sum()
        total_scp_increment = df["dSCP_g_protein"].sum()

        replacement_count = int(df["replacement_event"].sum())
        shutdown_count = int(df["shutdown_event"].sum())

        summary = {
            "initial_X_gDW_per_batch": self.X0_gdw,
            "initial_SCP_g_protein_per_batch": self.initial_scp_g,

            "final_active": bool(self.active),
            "final_X_gDW": self.X_gdw,
            "final_SCP_g_protein": self.SCP_g_protein,

            "total_SCP_increment_g_protein_before_harvest": total_scp_increment,
            "cumulative_harvested_SCP_g_protein": self.cumulative_harvested_scp_g,

            "total_H2_input_mol": df["H2_in_mol"].sum(),
            "total_H2_uptake_mol": total_h2_uptake,
            "total_H2_unused_mol": df["H2_unused_mol"].sum(),

            "total_H2_maintenance_mol": df["H2_maintenance_mol"].sum(),
            "total_H2_growth_mol": df["H2_growth_mol"].sum(),

            "average_H2_load_mol_h": df["H2_load_mol_h"].mean(),
            "average_H2_maintenance_fraction": df["H2_maintenance_fraction"].mean(),
            "final_H2_maintenance_fraction": df["H2_maintenance_fraction"].iloc[-1],

            "average_SCP_per_H2_uptake_g_per_mol": total_scp_increment / max(total_h2_uptake, 1e-12),

            "replacement_count": replacement_count,
            "shutdown_count": shutdown_count,

            "main_limiting_factor": df["limiting_factor"].mode()[0],
        }

        return pd.DataFrame([summary])

    def plot(self, df: Optional[pd.DataFrame] = None, figure_dir: str = "figure") -> None:
        """
        Plot main load-dispatch and biological-state trajectories.
        """
        if df is None:
            df = pd.DataFrame(self.records)

        if df.empty:
            raise ValueError("No records to plot")

        os.makedirs(figure_dir, exist_ok=True)

        def add_event_lines() -> None:
            for t in df.loc[df["replacement_event"], "time_h"]:
                plt.axvline(t, linestyle="--", linewidth=1, label="_replacement")

            for t in df.loc[df["shutdown_event"], "time_h"]:
                plt.axvline(t, linestyle=":", linewidth=1, label="_shutdown")

        # H2 load
        plt.figure()
        plt.plot(df["time_h"], df["H2_load_mol_h"], marker="o", label="Actual H2 load")
        plt.plot(df["time_h"], df["H2_input_mol_h"], linestyle="--", label="Available H2")
        add_event_lines()
        plt.xlabel("Time, h")
        plt.ylabel("H2, mol/h")
        plt.title("Dispatchable HOB hydrogen load")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "hob_h2_load.png"), dpi=300)
        plt.show()

        # Biomass
        plt.figure()
        plt.plot(df["time_h"], df["X_gDW"], marker="o", label="Biomass before event")
        plt.plot(df["time_h"], df["X_gDW_after_event"], marker="s", label="Biomass after event")
        add_event_lines()
        plt.xlabel("Time, h")
        plt.ylabel("Biomass, gDW")
        plt.title("HOB biomass trajectory")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "hob_biomass.png"), dpi=300)
        plt.show()

        # H2 allocation fractions
        plt.figure()
        plt.plot(df["time_h"], df["H2_maintenance_fraction"], marker="o", label="Maintenance fraction")
        plt.plot(df["time_h"], df["H2_growth_fraction"], marker="s", label="Growth fraction")
        plt.axhline(
            self.config.maintenance_fraction_cutoff,
            linestyle="--",
            label="Replacement cutoff",
        )
        add_event_lines()
        plt.xlabel("Time, h")
        plt.ylabel("Fraction of H2 uptake")
        plt.title("H2 allocation: maintenance vs growth")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "hob_h2_allocation.png"), dpi=300)
        plt.show()

        # Efficiency
        plt.figure()
        plt.plot(df["time_h"], df["SCP_per_H2_uptake_g_per_mol"], marker="o")
        add_event_lines()
        plt.xlabel("Time, h")
        plt.ylabel("g protein / mol H2")
        plt.title("Instantaneous SCP yield per H2 uptake")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "hob_scp_per_h2.png"), dpi=300)
        plt.show()


if __name__ == "__main__":
    params = HOBParams()

    config = HOBLoadConfig(
        maintenance_fraction_cutoff=0.5,
        min_runtime_before_replacement_h=24.0,
        auto_replace=True,
        restart_immediately_after_replacement=True,
        reset_on_shutdown=True,
    )

    hob_load = HOBHydrogenLoad(
        X0_gdw=5.0,
        params=params,
        load_config=config,
        start_active=True,
    )

    hours = 240
    dt_h = 1.0

    # Example: external scheduler shuts down this load from hour 90 to 120.
    load_on = np.ones(hours, dtype=bool)
    load_on[90:120] = False

    df = hob_load.simulate(
        F_H2_mol_h=0.20,
        F_O2_mol_h=0.08,
        F_CO2_mol_h=0.025,
        hours=hours,
        dt_h=dt_h,
        load_on_profile=load_on,
    )

    print("\nFirst rows:")
    print(df.head())

    print("\nReplacement / shutdown events:")
    event_cols = [
        "time_h",
        "cycle_id",
        "age_h",
        "replacement_event",
        "shutdown_event",
        "replacement_reason",
        "H2_maintenance_fraction",
        "harvested_SCP_increment_g_protein",
        "cumulative_harvested_SCP_g_protein",
    ]

    print(df.loc[df["replacement_event"] | df["shutdown_event"], event_cols])

    print("\nSummary:")
    print(hob_load.summarize(df))

    hob_load.plot(df, figure_dir="figure")