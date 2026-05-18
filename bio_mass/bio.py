from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

@dataclass
class HOBParams:
    """
    Literature-informed reduced HOB-SCP model.

    Target organism:
    - Hydrogen-oxidizing bacteria, e.g. Cupriavidus necator-like HOB.

    Units:
    - mu: h^-1
    - q: mmol / gDW / h
    - substrate_per_gdw: mmol substrate / gDW biomass

    Notes:
    - h2_per_gdw, o2_per_gdw, co2_per_gdw are derived from an empirical
      autotrophic biomass stoichiometry.
    - maintenance and capacity parameters are calibration parameters.
    """

    # ------------------------------------------------------------
    # More conservative autotrophic growth setting
    # ------------------------------------------------------------
    mu_max: float = 0.12          # h^-1, conservative autotrophic maximum
    f_protein: float = 0.65       # g protein / gDW biomass

    # ------------------------------------------------------------
    # Literature-informed biomass stoichiometry constraints
    # Approximate:
    # CO2 + 7.77 H2 + 2.87 O2 + 0.24 NH3
    # -> CH1.68O0.46N0.24 + 7.28 H2O
    #
    # Biomass MW per C-mol ≈ 24.4 gDW
    # ------------------------------------------------------------
    h2_per_gdw: float = 318.0     # mmol H2 / gDW biomass
    o2_per_gdw: float = 118.0     # mmol O2 / gDW biomass
    co2_per_gdw: float = 41.0     # mmol CO2 / gDW biomass

    # ------------------------------------------------------------
    # Maintenance parameters
    # These should be calibrated with experimental data.
    # ------------------------------------------------------------
    m_h2: float = 2.0             # mmol H2 / gDW / h
    m_o2: float = 0.75            # mmol O2 / gDW / h

    # ------------------------------------------------------------
    # Lumped biological capacity constraints
    # These are not directly validated enzyme constants.
    # They are chosen to be compatible with mu_max ≈ 0.12 h^-1.
    # They should be calibrated.
    # ------------------------------------------------------------
    q_h2_hydrogenase_cap: float = 45.0   # mmol H2 / gDW / h
    q_o2_etc_cap: float = 18.0           # mmol O2 / gDW / h
    q_co2_cbb_cap: float = 6.0           # mmol CO2 / gDW / h

    # Proteome / biosynthetic capacity cap
    mu_proteome_cap: float = 0.12        # h^-1

    # Starvation decay if maintenance cannot be met
    # Calibration parameter.
    k_starvation: float = 0.03           # h^-1 at full starvation


def _profile(x, n):
    """
    Convert scalar or array-like input to length-n numpy array.
    """
    if np.isscalar(x):
        return np.full(n, float(x))

    x = np.asarray(x, dtype=float)

    if len(x) != n:
        raise ValueError(f"Input profile length must be {n}, got {len(x)}")

    return x


def simulate_hob_scp(
    F_H2_mol_h,
    F_O2_mol_h,
    F_CO2_mol_h,
    X0_gdw,
    hours=240,
    dt_h=1.0,
    params=None,
):
    """
    Reduced dynamic metabolic demo for HOB -> SCP growth.

    This model does NOT include:
    - gas-liquid mass transfer
    - kLa
    - explicit gas or liquid phase
    - reactor hydrodynamics
    - pH dynamics
    - nitrogen limitation
    - mineral limitation

    It DOES include:
    - total hourly H2/O2/CO2 input
    - initial biomass
    - biomass growth
    - SCP protein accumulation
    - non-growth maintenance
    - H2 hydrogenase capacity
    - O2 respiratory chain capacity
    - CO2/CBB capacity
    - maximum autotrophic growth rate
    - proteome / biosynthetic cap
    - starvation if maintenance is not met

    Inputs
    ------
    F_H2_mol_h:
        H2 input rate, mol/h.
        Can be scalar or array with one value per time step.

    F_O2_mol_h:
        O2 input rate, mol/h.
        Can be scalar or array with one value per time step.

    F_CO2_mol_h:
        CO2 input rate, mol/h.
        Can be scalar or array with one value per time step.

    X0_gdw:
        Initial HOB biomass, gDW.

    hours:
        Simulation duration, h.

    dt_h:
        Time step, h.

    params:
        HOBParams object.

    Returns
    -------
    pandas.DataFrame:
        Time course of biomass, SCP, gas uptake, maintenance demand,
        growth demand, unused gas, growth rate, and limiting factor.
    """

    if params is None:
        params = HOBParams()

    if X0_gdw <= 0:
        raise ValueError("X0_gdw must be > 0")

    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    if hours <= 0:
        raise ValueError("hours must be > 0")

    n = int(np.ceil(hours / dt_h))

    H2_in = _profile(F_H2_mol_h, n)
    O2_in = _profile(F_O2_mol_h, n)
    CO2_in = _profile(F_CO2_mol_h, n)

    if np.any(H2_in < 0) or np.any(O2_in < 0) or np.any(CO2_in < 0):
        raise ValueError("Gas input rates must be non-negative")

    X = float(X0_gdw)
    SCP = params.f_protein * X

    rows = []

    for i in range(n):
        t = i * dt_h

        # ------------------------------------------------------------
        # 1. Gas input per unit biomass
        # ------------------------------------------------------------
        # No explicit gas/liquid phase.
        # Each hour's gas input is treated as the available gas pool
        # for that time step.
        #
        # mol/h -> mmol/gDW/h
        q_h2_supply = 1000.0 * H2_in[i] / max(X, 1e-12)
        q_o2_supply = 1000.0 * O2_in[i] / max(X, 1e-12)
        q_co2_supply = 1000.0 * CO2_in[i] / max(X, 1e-12)

        # ------------------------------------------------------------
        # 2. Biological capacity constraints
        # ------------------------------------------------------------
        # Even if input gas is high, the cell cannot use more than its
        # lumped enzyme/metabolic capacity.
        q_h2_available = min(q_h2_supply, params.q_h2_hydrogenase_cap)
        q_o2_available = min(q_o2_supply, params.q_o2_etc_cap)
        q_co2_available = min(q_co2_supply, params.q_co2_cbb_cap)

        h2_capacity_active = q_h2_supply > params.q_h2_hydrogenase_cap
        o2_capacity_active = q_o2_supply > params.q_o2_etc_cap
        co2_capacity_active = q_co2_supply > params.q_co2_cbb_cap

        # ------------------------------------------------------------
        # 3. Maintenance demand
        # ------------------------------------------------------------
        # Total maintenance demand increases as biomass increases.
        h2_maintenance_required_mol = X * params.m_h2 * dt_h / 1000.0
        o2_maintenance_required_mol = X * params.m_o2 * dt_h / 1000.0

        phi_maint = min(
            q_h2_available / params.m_h2 if params.m_h2 > 0 else 1.0,
            q_o2_available / params.m_o2 if params.m_o2 > 0 else 1.0,
            1.0,
        )

        # ------------------------------------------------------------
        # 4. Starvation case
        # ------------------------------------------------------------
        if phi_maint < 1.0:
            mu = 0.0
            starvation = 1.0 - phi_maint
            decay = params.k_starvation * starvation

            dX = -decay * X * dt_h
            dSCP = params.f_protein * dX

            h2_maintenance_mol = h2_maintenance_required_mol * phi_maint
            o2_maintenance_mol = o2_maintenance_required_mol * phi_maint

            h2_growth_mol = 0.0
            o2_growth_mol = 0.0
            co2_growth_mol = 0.0

            h2_uptake_mol = h2_maintenance_mol
            o2_uptake_mol = o2_maintenance_mol
            co2_uptake_mol = 0.0

            limiting = "maintenance/starvation"

        # ------------------------------------------------------------
        # 5. Growth case
        # ------------------------------------------------------------
        else:
            starvation = 0.0

            # Remaining specific gas availability after maintenance
            q_h2_growth_available = max(0.0, q_h2_available - params.m_h2)
            q_o2_growth_available = max(0.0, q_o2_available - params.m_o2)

            # Candidate growth rates from constraints.
            # Growth is controlled by the most restrictive constraint.
            mu_candidates = {
                "H2": q_h2_growth_available / params.h2_per_gdw,
                "O2": q_o2_growth_available / params.o2_per_gdw,
                "CO2/CBB": q_co2_available / params.co2_per_gdw,
                "mu_max": params.mu_max,
                "proteome": params.mu_proteome_cap,
            }

            limiting = min(mu_candidates, key=mu_candidates.get)
            mu = max(0.0, mu_candidates[limiting])

            dX = mu * X * dt_h
            dSCP = params.f_protein * dX

            # Maintenance consumption
            h2_maintenance_mol = h2_maintenance_required_mol
            o2_maintenance_mol = o2_maintenance_required_mol

            # Growth-associated consumption
            h2_growth_mol = X * params.h2_per_gdw * mu * dt_h / 1000.0
            o2_growth_mol = X * params.o2_per_gdw * mu * dt_h / 1000.0
            co2_growth_mol = X * params.co2_per_gdw * mu * dt_h / 1000.0

            h2_uptake_mol = h2_maintenance_mol + h2_growth_mol
            o2_uptake_mol = o2_maintenance_mol + o2_growth_mol
            co2_uptake_mol = co2_growth_mol

        # ------------------------------------------------------------
        # 6. Numerical guard: cannot consume more than supplied
        # ------------------------------------------------------------
        h2_uptake_mol = min(h2_uptake_mol, H2_in[i] * dt_h)
        o2_uptake_mol = min(o2_uptake_mol, O2_in[i] * dt_h)
        co2_uptake_mol = min(co2_uptake_mol, CO2_in[i] * dt_h)

        # ------------------------------------------------------------
        # 7. Fractions
        # ------------------------------------------------------------
        h2_maintenance_fraction = h2_maintenance_mol / max(h2_uptake_mol, 1e-12)
        h2_growth_fraction = h2_growth_mol / max(h2_uptake_mol, 1e-12)

        o2_maintenance_fraction = o2_maintenance_mol / max(o2_uptake_mol, 1e-12)
        o2_growth_fraction = o2_growth_mol / max(o2_uptake_mol, 1e-12)

        h2_maintenance_fraction = min(max(h2_maintenance_fraction, 0.0), 1.0)
        h2_growth_fraction = min(max(h2_growth_fraction, 0.0), 1.0)

        o2_maintenance_fraction = min(max(o2_maintenance_fraction, 0.0), 1.0)
        o2_growth_fraction = min(max(o2_growth_fraction, 0.0), 1.0)

        # ------------------------------------------------------------
        # 8. Update state
        # ------------------------------------------------------------
        X = max(0.0, X + dX)
        SCP = max(0.0, SCP + dSCP)

        rows.append({
            "time_h": t + dt_h,

            # Biomass and SCP
            "X_gDW": X,
            "SCP_g_protein": SCP,
            "dX_gDW": dX,
            "dSCP_g_protein": dSCP,
            "mu_h-1": mu,
            "limiting_factor": limiting,
            "starvation_level": starvation,

            # Gas input
            "H2_in_mol": H2_in[i] * dt_h,
            "O2_in_mol": O2_in[i] * dt_h,
            "CO2_in_mol": CO2_in[i] * dt_h,

            # Specific supply
            "q_H2_supply_mmol_gDW_h": q_h2_supply,
            "q_O2_supply_mmol_gDW_h": q_o2_supply,
            "q_CO2_supply_mmol_gDW_h": q_co2_supply,

            # Specific biologically available uptake capacity
            "q_H2_available_mmol_gDW_h": q_h2_available,
            "q_O2_available_mmol_gDW_h": q_o2_available,
            "q_CO2_available_mmol_gDW_h": q_co2_available,

            # Whether capacity was active
            "H2_hydrogenase_cap_active": h2_capacity_active,
            "O2_ETC_cap_active": o2_capacity_active,
            "CO2_CBB_cap_active": co2_capacity_active,

            # Actual uptake
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
            "H2_unused_mol": max(0.0, H2_in[i] * dt_h - h2_uptake_mol),
            "O2_unused_mol": max(0.0, O2_in[i] * dt_h - o2_uptake_mol),
            "CO2_unused_mol": max(0.0, CO2_in[i] * dt_h - co2_uptake_mol),
        })

    return pd.DataFrame(rows)


def summarize_result(df, X0_gdw, params):
    """
    Return summary statistics as a one-row DataFrame.
    """

    initial_scp = params.f_protein * X0_gdw
    final_scp = df["SCP_g_protein"].iloc[-1]
    final_x = df["X_gDW"].iloc[-1]

    summary = {
        "initial_X_gDW": X0_gdw,
        "final_X_gDW": final_x,
        "initial_SCP_g_protein": initial_scp,
        "final_SCP_g_protein": final_scp,
        "SCP_increment_g_protein": final_scp - initial_scp,

        "total_H2_input_mol": df["H2_in_mol"].sum(),
        "total_H2_uptake_mol": df["H2_uptake_mol"].sum(),
        "total_H2_unused_mol": df["H2_unused_mol"].sum(),
        "total_H2_maintenance_mol": df["H2_maintenance_mol"].sum(),
        "total_H2_growth_mol": df["H2_growth_mol"].sum(),

        "total_O2_input_mol": df["O2_in_mol"].sum(),
        "total_O2_uptake_mol": df["O2_uptake_mol"].sum(),

        "total_CO2_input_mol": df["CO2_in_mol"].sum(),
        "total_CO2_uptake_mol": df["CO2_uptake_mol"].sum(),

        "final_mu_h-1": df["mu_h-1"].iloc[-1],
        "final_H2_maintenance_fraction": df["H2_maintenance_fraction"].iloc[-1],
        "main_limiting_factor": df["limiting_factor"].mode()[0],
    }

    return pd.DataFrame([summary])


def plot_results(df, figure_dir="figure"):
    """
    Plot and save main simulation results to figure folder.
    """

    os.makedirs(figure_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. SCP growth
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(df["time_h"], df["SCP_g_protein"], marker="o")
    plt.xlabel("Time, h")
    plt.ylabel("SCP, g protein")
    plt.title("Predicted SCP growth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_scp_growth.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 2. Biomass growth
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(df["time_h"], df["X_gDW"], marker="o")
    plt.xlabel("Time, h")
    plt.ylabel("HOB biomass, gDW")
    plt.title("HOB biomass growth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_biomass_growth.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 3. Specific growth rate
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(df["time_h"], df["mu_h-1"], marker="o")
    plt.xlabel("Time, h")
    plt.ylabel("Specific growth rate, h$^{-1}$")
    plt.title("Specific growth rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_specific_growth_rate.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 4. Gas uptake
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(df["time_h"], df["H2_uptake_mol"], marker="o", label="H2 uptake")
    plt.plot(df["time_h"], df["O2_uptake_mol"], marker="s", label="O2 uptake")
    plt.plot(df["time_h"], df["CO2_uptake_mol"], marker="^", label="CO2 uptake")
    plt.xlabel("Time, h")
    plt.ylabel("Gas uptake, mol per step")
    plt.title("Actual gas uptake per hour")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_gas_uptake.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 5. H2 maintenance vs growth
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(
        df["time_h"],
        df["H2_maintenance_mol"],
        marker="o",
        label="H2 maintenance",
    )
    plt.plot(
        df["time_h"],
        df["H2_growth_mol"],
        marker="s",
        label="H2 growth",
    )
    plt.xlabel("Time, h")
    plt.ylabel("H2 consumption, mol per step")
    plt.title("H2 split: maintenance vs growth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_h2_maintenance_vs_growth.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 6. H2 allocation fraction
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(
        df["time_h"],
        df["H2_maintenance_fraction"],
        marker="o",
        label="Maintenance fraction",
    )
    plt.plot(
        df["time_h"],
        df["H2_growth_fraction"],
        marker="s",
        label="Growth fraction",
    )
    plt.xlabel("Time, h")
    plt.ylabel("Fraction of H2 uptake")
    plt.title("H2 allocation fraction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_h2_allocation_fraction.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 7. Unused gas
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(df["time_h"], df["H2_unused_mol"], marker="o", label="Unused H2")
    plt.plot(df["time_h"], df["O2_unused_mol"], marker="s", label="Unused O2")
    plt.plot(df["time_h"], df["CO2_unused_mol"], marker="^", label="Unused CO2")
    plt.xlabel("Time, h")
    plt.ylabel("Unused gas, mol per step")
    plt.title("Unused gas per hour")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "baseline_unused_gas.png"), dpi=300)
    plt.show()

def plot_h2_comparison(results, figure_dir="figure"):
    """
    Plot and save growth curves under different H2 input rates.
    """

    os.makedirs(figure_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Biomass growth under different H2 supplies
    # ------------------------------------------------------------
    plt.figure()

    for h2, df_i in results.items():
        plt.plot(
            df_i["time_h"],
            df_i["X_gDW"],
            marker="o",
            label=f"H2 = {h2} mol/h",
        )

    plt.xlabel("Time, h")
    plt.ylabel("HOB biomass, gDW")
    plt.title("HOB biomass growth under different H2 supplies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "comparison_h2_biomass_growth.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 2. SCP growth under different H2 supplies
    # ------------------------------------------------------------
    plt.figure()

    for h2, df_i in results.items():
        plt.plot(
            df_i["time_h"],
            df_i["SCP_g_protein"],
            marker="o",
            label=f"H2 = {h2} mol/h",
        )

    plt.xlabel("Time, h")
    plt.ylabel("SCP, g protein")
    plt.title("SCP growth under different H2 supplies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "comparison_h2_scp_growth.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 3. Specific growth rate under different H2 supplies
    # ------------------------------------------------------------
    plt.figure()

    for h2, df_i in results.items():
        plt.plot(
            df_i["time_h"],
            df_i["mu_h-1"],
            marker="o",
            label=f"H2 = {h2} mol/h",
        )

    plt.xlabel("Time, h")
    plt.ylabel("Specific growth rate, h$^{-1}$")
    plt.title("Specific growth rate under different H2 supplies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "comparison_h2_specific_growth_rate.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 4. H2 maintenance fraction
    # ------------------------------------------------------------
    plt.figure()

    for h2, df_i in results.items():
        plt.plot(
            df_i["time_h"],
            df_i["H2_maintenance_fraction"],
            marker="o",
            label=f"H2 = {h2} mol/h",
        )

    plt.xlabel("Time, h")
    plt.ylabel("H2 maintenance fraction")
    plt.title("Fraction of H2 used for maintenance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "comparison_h2_maintenance_fraction.png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 5. H2 uptake under different H2 supplies
    # ------------------------------------------------------------
    plt.figure()

    for h2, df_i in results.items():
        plt.plot(
            df_i["time_h"],
            df_i["H2_uptake_mol"],
            marker="o",
            label=f"H2 = {h2} mol/h",
        )

    plt.xlabel("Time, h")
    plt.ylabel("H2 uptake, mol per hour")
    plt.title("Actual H2 uptake under different H2 supplies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "comparison_h2_actual_uptake.png"), dpi=300)
    plt.show()

def compare_h2_supply(
    H2_values_mol_h,
    F_O2_mol_h,
    F_CO2_mol_h,
    X0_gdw,
    hours=240,
    dt_h=1.0,
    params=None,
):
    """
    Compare HOB biomass and SCP growth under different H2 input rates.

    Inputs
    ------
    H2_values_mol_h:
        List of H2 input rates, mol/h.

    F_O2_mol_h:
        Fixed O2 input rate, mol/h.

    F_CO2_mol_h:
        Fixed CO2 input rate, mol/h.

    X0_gdw:
        Initial HOB biomass, gDW.

    hours:
        Simulation duration, h.

    dt_h:
        Time step, h.

    params:
        HOBParams object.

    Returns
    -------
    dict:
        key = H2 input rate
        value = simulation dataframe

    pandas.DataFrame:
        summary table of final biomass, SCP, and limiting factors.
    """

    if params is None:
        params = HOBParams()

    results = {}
    summary_rows = []

    for h2 in H2_values_mol_h:
        df_i = simulate_hob_scp(
            F_H2_mol_h=h2,
            F_O2_mol_h=F_O2_mol_h,
            F_CO2_mol_h=F_CO2_mol_h,
            X0_gdw=X0_gdw,
            hours=hours,
            dt_h=dt_h,
            params=params,
        )

        results[h2] = df_i

        initial_scp = params.f_protein * X0_gdw
        final_scp = df_i["SCP_g_protein"].iloc[-1]
        final_x = df_i["X_gDW"].iloc[-1]

        summary_rows.append({
            "H2_mol_h": h2,
            "final_X_gDW": final_x,
            "final_SCP_g_protein": final_scp,
            "SCP_increment_g_protein": final_scp - initial_scp,
            "total_H2_input_mol": df_i["H2_in_mol"].sum(),
            "total_H2_uptake_mol": df_i["H2_uptake_mol"].sum(),
            "total_H2_unused_mol": df_i["H2_unused_mol"].sum(),
            "final_mu_h-1": df_i["mu_h-1"].iloc[-1],
            "final_H2_maintenance_fraction": df_i["H2_maintenance_fraction"].iloc[-1],
            "main_limiting_factor": df_i["limiting_factor"].mode()[0],
        })

    summary_df = pd.DataFrame(summary_rows)

    return results, summary_df


if __name__ == "__main__":
    params = HOBParams()

    # ------------------------------------------------------------
    # Baseline simulation
    # ------------------------------------------------------------
    df = simulate_hob_scp(
        F_H2_mol_h=0.20,
        F_O2_mol_h=0.08,
        F_CO2_mol_h=0.025,
        X0_gdw=5.0,
        hours=240,
        dt_h=1.0,
        params=params,
    )

    print("\nBaseline first rows:")
    print(df.head())

    print("\nBaseline last rows:")
    print(df.tail())

    print("\nBaseline summary:")
    print(summarize_result(df, X0_gdw=5.0, params=params))

    print("\nBaseline limiting factor counts:")
    print(df["limiting_factor"].value_counts())

    plot_results(df, figure_dir="figure")

    # ------------------------------------------------------------
    # H2 supply comparison
    # ------------------------------------------------------------
    H2_values = [0.05, 0.10, 0.20, 0.40, 0.80]

    h2_results, h2_summary = compare_h2_supply(
        H2_values_mol_h=H2_values,
        F_O2_mol_h=0.08,
        F_CO2_mol_h=0.025,
        X0_gdw=5.0,
        hours=240,
        dt_h=1.0,
        params=params,
    )

    print("\nH2 supply comparison summary:")
    print(h2_summary)

    plot_h2_comparison(h2_results, figure_dir="figure")
    