# Physics-Based Virtual Flow Meter (VFM) Mathematical & Interface Specification

**Scope:** Target mathematical formulation, unit conventions, edge-case rules, and module API contracts for upgrading `vfm_project/` from a pooled 26-feature `HybridVFM` regression into a mechanistic, per-well calibrated, inverse-variance reconciled physics VFM (`reservoir.py`, `choke.py`, `vlp.py`, `physics_vfm.py`, and scripts `05`–`08`).

---

## 1. System Overview & Physical Regime

Across all 10 gas-lifted wells (`GL-01` through `GL-10`, Jan–Sep 2025):
- **Reservoir regime:** Measured downhole gauge pressures (`2269–3914 psia`) exceed bubble-point pressures (`1177–1595 psia`) by at least `+1004 psi` on every well (`0%` of flowing minutes below $P_b$). Reservoir inflow is **single-phase undersaturated**, meaning linear Productivity Index ($J$) holds and producing formation GOR equals solution GOR ($R_{sb}$).
- **Elevation offset:** Reservoir datum TVD (`datum_tvd_ft`) is deeper than gauge TVD (`gauge_tvd_ft`) by `+150 ft` to `+420 ft`, adding `+54` to `+170 psi` of hydrostatic head between datum and gauge.
- **Choke regime:** `choke_opening_pct` maps linearly to **bean diameter** (`choke_max_size_64ths = 128`, i.e., $d_{\max} = 2.0\text{ in}$). Downstream-to-upstream pressure ratio `whp_downstream_psia / whp_psia` spans both critical (`<= 0.55`: `GL-04` 98.5%, `GL-10` 86.7%, `GL-06` 67.4%, `GL-09` 29.4%) and subcritical (`> 0.55`: `GL-01, 02, 03, 05, 07, 08` 95–100%).
- **Tubing architecture:** Lift gas enters at `gl_valve_tvd_ft` (`2680–5770 ft`), strictly above `gauge_tvd_ft` (`7450–9850 ft`). Below the valve, only formation gas ($R_{sb}$) is present (entirely in solution while $P > P_b$); above the valve, both formation gas and injected lift gas expand up to `whp_psia`.

---

## 2. Standard Units & Physical Constants

| Symbol | Quantity | Unit | Constant / Conversion |
|---|---|---|---|
| $q_o, q_w, q_{\text{liq}}$ | Oil, water, liquid rates | $\text{STB/d}$ | $1\text{ STB} = 5.614583\text{ ft}^3$ |
| $q_{g,\text{form}}, q_{\text{gl}}, q_{g,\text{tot}}$ | Formation, lift, total gas rates | $\text{Mscf/d}$ | $1\text{ Mscf/d} = 1000\text{ scf/d} = 28.3168466\text{ Sm}^3/\text{d}$ |
| $GOR, GLR, R_s$ | Gas-oil / gas-liquid ratio | $\text{scf/STB}$ | $GOR = 1000\,q_{g,\text{form}} / q_o$ |
| $WC$ | Water cut fraction | $[-]$ in $[0, 1]$ | $WC = q_w / q_{\text{liq}} = \text{water\_cut\_pct}/100$ |
| $P, \Delta P$ | Pressure / differential pressure | $\text{psia}$, $\text{psi}$ | $1\text{ psi} = 144\text{ lb}_f/\text{ft}^2$ |
| $T, T_R$ | Temperature | $^\circ\text{F}$, $^\circ\text{R}$ | $T_R = T_{^\circ\text{F}} + 459.67$ |
| $d, ID$ | Choke bean / tubing inner diameter | $\text{in}$ | $d_{\text{ft}} = d_{\text{in}} / 12$ |
| $z$ | True Vertical Depth (TVD) | $\text{ft}$ | Positive downward |
| $J$ | Productivity Index | $\text{STB/(d}\cdot\text{psi)}$ | Undersaturated liquid PI |
| $g_c, R$ | Gravitational conversion / gas constant | — | $g_c = 32.174$, $R = 10.7316\text{ psia}\cdot\text{ft}^3/(\text{lbmol}\cdot^\circ\text{R})$ |
| $\rho_{\text{air,std}}$ | Air density at $14.7\text{ psia}, 60^\circ\text{F}$ | $\text{lb}_m/\text{scf}$ | $\rho_{\text{air,std}} = 0.07636\text{ lb}_m/\text{scf}$ ($\rho_{g,\text{std}} = 0.07636\,\gamma_g$) |
| $\gamma_o, \gamma_w$ | Oil / water specific gravity | $[-]$ | $\gamma_o = \frac{141.5}{131.5 + \text{API}}$, $\gamma_w = 1 + 0.695\times 10^{-6}\cdot S_{\text{ppm}}$ |

---

## 3. Component Equations

### 3.1 (a) Gauge-to-Datum Pressure, Build-up Analysis (PBU), and Undersaturated/Composite IPR (`reservoir.py`)

#### Wellbore Fluid Gradient & Datum Conversion
Because `gauge_tvd_ft` is deeper than `gl_valve_tvd_ft` and $P_{dh} > P_b$, the interval `[gauge_tvd_ft, datum_tvd_ft]` contains single-phase undersaturated liquid (live oil with dissolved gas $R_s = R_{sb}$ plus brine):
$$\rho_{o,\text{bh}} = \frac{62.428\,\gamma_o + 0.01361\,R_s\,\gamma_g}{B_o}, \qquad \rho_{w,\text{bh}} = \frac{62.428\,\gamma_w}{B_w} \quad [\text{lb}_m/\text{ft}^3]$$
$$\rho_{L,\text{bh}} = (1 - WC)\,\rho_{o,\text{bh}} + WC\,\rho_{w,\text{bh}}, \qquad \gamma_{\text{bh}} = \frac{\rho_{L,\text{bh}}}{144.0} \quad [\text{psi/ft}]$$
Conversion from gauge TVD $z_g$ to reservoir datum TVD $z_d$:
$$P_{\text{datum}} = P_{\text{gauge}} + \gamma_{\text{bh}} \cdot (z_d - z_g)$$
*(Verify sign: since $z_d > z_g$ on all 10 wells, $P_{\text{datum}} > P_{\text{gauge}}$ by $\sim 54\text{--}170\text{ psi}$.)*

#### Pressure Build-Up (PBU) Extraction & $P_r$ Anchor Estimation
1. **True shut-in identification:** Detect contiguous shut-in intervals ONLY where **`choke_opening_pct <= 0.5`** (do NOT trigger on `dp_choke <= 2.0` alone, which causes false 6-minute spikes such as `GL-03` on `2025-09-06 08:20` at `50%` choke).
2. **Shut-in time-zero alignment:** In `high_freq_data.csv`, the row at event timestamp $t_0$ still carries the final flowing downhole pressure $P_{wf}(\Delta t = 0)$, and build-up begins at $t_0 + 1\text{ min}$ ($\Delta t_i = (t_i - t_0)\text{ hr} > 0$).
3. **Horner / Agarwal equivalent time & Bourdet derivative:**
   - Prior producing rate $q_{\text{prior}}$ ($\text{STB/d}$) is taken from the nearest causal prior well test ($t_{\text{test\_end}} \le t_0$) or effective flowing rate; producing time $t_p$ is estimated from cumulative oil or time since previous shut-in (capped safely, e.g. $t_p \ge 240\text{ hr}$ when continuous).
   - Horner time: $X_H = \frac{t_p + \Delta t}{\Delta t}$ (and Agarwal equivalent time $\Delta t_e = \frac{t_p \Delta t}{t_p + \Delta t}$).
   - Bourdet logarithmic derivative: $\Delta p'(\Delta t_i) = \frac{d(p_{ws} - p_{wf,0})}{d\ln \Delta t}$ computed on log-spaced bins (window $L \approx 0.15\text{--}0.25$) to identify the radial-flow middle-time region (flat $\Delta p'$) vs early wellbore storage (unit slope).
   - **Extrapolated $p^*$:** Fit late-time Horner straight line $p_{ws}(\Delta t) = p^* - m_H \log_{10}\!\left(\frac{t_p + \Delta t}{\Delta t}\right)$ over the late-time radial window (e.g. $\Delta t$ in upper 50% of build-up after storage, enforcing $p^* \ge p_{\text{end}}$ and bounding overshoot on short shut-ins like `GL-02` 5.6h, `GL-06` 8h, `GL-10` 9h). Convert $p^*$ to datum using `gauge_to_datum_pressure`.
   - Diagnostic reservoir properties (when clear radial plateau $\Delta p'_{\text{rad}}$ exists):
     $$k h = \frac{162.6 \, q_{\text{prior}} B_L \mu_L}{m_H} \quad [\text{md}\cdot\text{ft}], \qquad s = 1.151 \left[\frac{p_{1\text{hr}} - p_{wf,0}}{m_H} - \log_{10}\!\left(\frac{k}{\phi \mu_L c_t r_w^2}\right) + 3.23\right].$$

#### Reservoir Pressure Trend $P_r(t)$ & Productivity Index $J$
All pressures are evaluated at `datum_tvd_ft`:
- Linear reservoir pressure decline model per well:
  $$P_r(t) = P_{r0} - a \cdot t_{\text{days}}, \qquad a \ge 0, \quad J > 0, \quad P_r(t_k) > P_{wf,k}$$
  where $t_{\text{days}} = (t - t_{\text{ref}}) / 86400\text{ s}$ with $t_{\text{ref}} = \text{2025-01-01T00:00:00Z}$.
- **Wells with causal PBU anchors (`t_pbu <= max_time`):** Anchor $P_r(t_{\text{pbu}}) = P_{r,\text{pbu}}$ (weighted strongly by build-up quality/duration: 48–72h shut-ins have high precision; 5.6–9h shut-ins have wider tolerance) and jointly fit $(P_{r0}, a, J)$ minimizing test liquid-rate errors plus anchor residual (with non-negative decline prior $a \ge 0$).
- **Wells without causal PBU anchors (or prior to first shut-in during walk-forward CV):** Jointly fit $(P_{r0}, a, J)$ from well-test equations $q_{\text{liq},k} = J\,(P_{r0} - a\,t_k - P_{wf,k})$ with weak regularization on $a$ when $n_{\text{tests}} \le 3$ so $(P_{r0}, a, J)$ stays identifiable and $P_r(t) > P_{wf}$.

#### Undersaturated IPR & Smooth Composite Vogel Switch
Given $J > 0$, $P_r$, $P_{wf}$ (both at datum TVD), and $P_b$:
1. If $P_{wf} \ge P_r$: $q_{\text{liq}} = 0$.
2. If $P_b$ is `None` or $P_r > P_b$ and $P_{wf} \ge P_b$ (undersaturated linear regime):
   $$q_{\text{liq}}(P_{wf}) = J\,(P_r - P_{wf}).$$
3. If $P_r > P_b$ and $0 \le P_{wf} < P_b$ (composite Vogel below bubble point, $C^1$-continuous at $P_b$):
   $$q_b = J\,(P_r - P_b), \qquad q_{\text{liq}}(P_{wf}) = q_b + \frac{J\,P_b}{1.8}\left[1 - 0.2\left(\frac{P_{wf}}{P_b}\right) - 0.8\left(\frac{P_{wf}}{P_b}\right)^2\right].$$
   *Continuity proof at $P_{wf} = P_b$:* $q(P_b^-) = q_b = q(P_b^+)$, and $\left.\frac{dq}{dP_{wf}}\right|_{P_b^-} = \frac{J P_b}{1.8}\left(-\frac{0.2}{P_b} - \frac{1.6}{P_b}\right) = -J = \left.\frac{dq}{dP_{wf}}\right|_{P_b^+}$.
4. If $P_r \le P_b$ (saturated reservoir):
   $$q_{\text{liq}}(P_{wf}) = \frac{J\,P_r}{1.8}\left[1 - 0.2\left(\frac{P_{wf}}{P_r}\right) - 0.8\left(\frac{P_{wf}}{P_r}\right)^2\right].$$

---

### 3.2 (b) Multiphase Choke Flow Model (`choke.py`)

#### Bean Diameter Mapping
Empirical analysis of `choke_events.csv` proves `choke_opening_pct` represents discrete stem steps of `bean_64ths / choke_max_size_64ths * 100` (all 22 unique values map to even 128ths, i.e., integer 32nds of an inch out of `128/64 = 2.0 in`). Thus `choke_opening_pct` is **linear in bean diameter**:
$$d_{\text{in}} = \left(\frac{\text{clip}(\text{pct}, 0, 100)}{100}\right) \cdot \left(\frac{\text{choke\_max\_size\_64ths}}{64}\right), \qquad A_{\text{choke}} = \frac{\pi}{4}\left(\frac{d_{\text{in}}}{12}\right)^2 \quad [\text{ft}^2].$$

#### Critical / Subcritical Transition & Homogeneous Orifice Equation
Let $P_{\text{wh}} = \text{whp\_psia}$, $P_{\text{ds}} = \text{whp\_downstream\_psia}$, pressure ratio $r = P_{\text{ds}} / P_{\text{wh}}$, and critical pressure ratio $r_c = 0.55$ (classic Gilbert/Ros/Sachdeva multiphase critical transition). Define the effective choked pressure drop:
$$\Delta P_{\text{eff}} = \begin{cases} 0, & \text{if } d_{\text{in}} \le 0 \text{ or } P_{\text{wh}} \le P_{\text{ds}} \text{ or } \text{pct} \le 0.5, \\ P_{\text{wh}} - P_{\text{ds}}, & \text{if } r_c < r < 1, \\ (1 - r_c)\,P_{\text{wh}}, & \text{if } r \le r_c. \end{cases}$$

At upstream conditions $(P_{\text{wh}}, T_{\text{wh}})$ where $T_{\text{wh}} \approx T_{\text{sep}} + 25^\circ\text{F}$ (or per-well typical $T_{\text{wh}} \approx 120^\circ\text{F} = 579.67^\circ\text{R}$ when separator temperature is absent in 1-min telemetry):
- Solution gas still dissolved at $P_{\text{wh}}$ via Standing correlation:
  $$R_{s,\text{wh}} = \min\!\left(R_{sb},\; \gamma_g \left[\left(\frac{P_{\text{wh}}}{18.2} + 1.4\right)\,10^{0.0125\,\text{API} - 0.00091\,T_{\text{wh}}}\right]^{1.2048}\right).$$
- Total gas-liquid ratio $\text{GLR}_{\text{tot}} = (1 - WC)\,GOR + \frac{1000\,q_{\text{gl}}}{q_{\text{liq}}}$ ($\text{scf/STB}$), with free gas at choke:
  $$\text{GLR}_{\text{free}} = \max\!\left(\text{GLR}_{\text{tot}} - (1 - WC)\,R_{s,\text{wh}},\; 0\right).$$
- Gas formation volume factor $B_g(P_{\text{wh}}, T_{\text{wh}}) = 0.02827\,\frac{Z(P_{\text{wh}}, T_{\text{wh}})\,T_{\text{wh},R}}{P_{\text{wh}}}$ ($\text{ft}^3/\text{scf}$), oil/water liquid FVF $B_L \approx (1 - WC)B_{o,\text{wh}} + WC\,B_w$.
- Total mass and in-situ volume per STB of stock-tank liquid:
  $$m_{\text{tot}} = 5.614583\,\rho_{L,\text{std}} + \text{GLR}_{\text{tot}}\,(0.07636\,\gamma_{g,\text{mix}}) \quad [\text{lb}_m/\text{STB}]$$
  $$V_{\text{mix}} = 5.614583\,B_L + \text{GLR}_{\text{free}}\,B_g \quad [\text{ft}^3/\text{STB}], \qquad \rho_{\text{mix}} = \frac{m_{\text{tot}}}{V_{\text{mix}}} \quad [\text{lb}_m/\text{ft}^3].$$
- Theoretical homogeneous orifice mass flow and calibrated well liquid rate:
  $$\dot{m}_{\text{theo}} = 86400 \, A_{\text{choke}} \sqrt{2 g_c \cdot 144 \cdot \Delta P_{\text{eff}} \cdot \rho_{\text{mix}}} \quad [\text{lb}_m/\text{d}],$$
  $$q_{\text{liq,choke}} = C_{d,w} \cdot \frac{\dot{m}_{\text{theo}}}{m_{\text{tot}}} \quad [\text{STB/d}].$$
*(Physical check: across all 85 well tests, $C_{d,w} \in [0.75, 0.85]$ with $\le 3\%$ scatter, matching textbook multiphase discharge coefficients!)*

#### Fixed-Point Solution with Known $q_{\text{gl}}$ (1-min Inference)
During 1-minute prediction, $q_{\text{gl}}$ ($\text{Mscf/d}$) is measured while $\text{GLR}_{\text{tot}}$ depends on $q_{\text{liq}}$. Solve $q_{\text{liq}} = \mathcal{F}_{\text{choke}}(q_{\text{liq}};\, P_{\text{wh}}, P_{\text{ds}}, d_{\text{in}}, q_{\text{gl}}, WC, GOR, C_{d,w})$ via 4–5 damped fixed-point iterations initialized at $q_{\text{liq}}^{(0)}$ (using well mean $q_{\text{liq}}$ or initial no-lift-gas choke pass), keeping strict monotonicity with $d_{\text{in}}$ and $\Delta P_{\text{eff}}$ and returning `0.0` identically when $\Delta P_{\text{eff}} \le 0$.

---

### 3.3 (c) Mechanistic Two-Segment Tubing VLP (`vlp.py`)

#### Two-Segment Geometry
Divide tubing at `gl_valve_tvd_ft` ($z_v$):
1. **Upper segment** $z \in [0, z_v]$ (discretized into $N_1 \ge 6$ steps of $\Delta z_1 = z_v / N_1$): carries oil, water, formation gas ($q_o \cdot GOR$), **and** injected lift gas ($1000\,q_{\text{gl}}$). Gas gravity is flow-weighted blend of formation gas (`gas_sg`) and lift gas (`lift_gas_sg`).
2. **Lower segment** $z \in [z_v, z_g]$ (discretized into $N_2 \ge 6$ steps of $\Delta z_2 = (z_g - z_v) / N_2$): carries oil, water, and formation gas **only** ($q_o \cdot GOR$).

#### Local Step Pressure Gradient (Top-Down Marching from $P_0 = P_{\text{wh}}$)
At depth step midpoint $(P_{\text{mid}}, T_{\text{mid}})$ with linear temperature profile $T(z) = T_{\text{wh}} + (T_{\text{res}} - T_{\text{wh}})\,(z / z_g)$:
1. **Real-gas Z-factor (Papay):**
   $$P_{pc} = 756.8 - 131.0\,\gamma_g - 3.6\,\gamma_g^2, \qquad T_{pc} = 169.2 + 349.5\,\gamma_g - 74.0\,\gamma_g^2,$$
   $$P_{pr} = \frac{P_{\text{mid}}}{P_{pc}}, \quad T_{pr} = \frac{T_{\text{mid},R}}{T_{pc}}, \qquad Z = \text{clip}\!\left(1 - \frac{3.52\,P_{pr}}{10^{0.9813\,T_{pr}}} + \frac{0.274\,P_{pr}^2}{10^{0.8157\,T_{pr}}},\; 0.25,\; 1.30\right).$$
2. **In-situ phase volumetric fluxes ($A_t = \frac{\pi}{4}(ID_{\text{ft}})^2$):**
   Using local $R_s(P_{\text{mid}}, T_{\text{mid}}) \le GOR$, $B_g(P_{\text{mid}}, T_{\text{mid}})$, $B_o$, $B_w$:
   $$q_{g,\text{free}} = \max\!\left(q_{g,\text{seg}} - q_o\,R_s(P_{\text{mid}}, T_{\text{mid}}),\; 0\right),$$
   $$v_{sl} = \frac{5.614583\,(q_o B_o + q_w B_w)}{86400\,A_t}, \qquad v_{sg} = \frac{q_{g,\text{free}}\,B_g}{86400\,A_t}, \qquad v_m = v_{sl} + v_{sg}.$$
3. **Drift-flux liquid holdup (with calibrated slip profile parameter $C_{0,w}$):**
   $$\alpha_g = \begin{cases} 0 & \text{if } q_{g,\text{free}} \le 0, \\ \text{clip}\!\left(\dfrac{v_{sg}}{C_{0,w}\,v_m + v_d},\; 0,\; 0.98\right) & \text{otherwise}, \end{cases} \qquad H_L = 1 - \alpha_g$$
   where $v_d = 0.65\text{ ft/s}$ (Harmathy churn/bubble drift proxy) and $C_{0,w} \in [1.0, 1.6]$ is calibrated per well.
   $$\rho_m = H_L\,\rho_L + (1 - H_L)\,\rho_g.$$
4. **Darcy-Weisbach friction (Haaland explicit friction factor, calibrated multiplier $f_{\text{mult},w}$):**
   $$\text{Re} = \frac{\rho_m\,v_m\,ID_{\text{ft}}}{\mu_m}, \qquad f_D = \left[-1.8\log_{10}\!\left(\left(\frac{\varepsilon / ID}{3.7}\right)^{1.11} + \frac{6.9}{\text{Re}}\right)\right]^{-2}$$
   with default pipe roughness $\varepsilon = 0.0006\text{ in}$, and total step pressure rise:
   $$\Delta P_{\text{step}} = \frac{\Delta z}{144.0}\left[\rho_m + f_{\text{mult},w}\,f_D\,\frac{\rho_m\,v_m^2}{2 g_c\,ID_{\text{ft}}}\right].$$
   Two predictor-corrector iterations per depth step align $P_{\text{mid}} = P + \frac{1}{2}\Delta P_{\text{step}}$ cleanly.

#### Numerical Inversion for $q_{\text{liq}}$ & Conditioning Guard
Given measured $\Delta P_{\text{meas}} = P_{dh} - P_{\text{wh}}$, $WC$, $P_{\text{wh}}$, $q_{\text{gl}}$:
- Search $q_{\text{liq}} \in [q_{\min}, q_{\max}]$ (e.g., $[50, 12000]\text{ STB/d}$).
- Check bracketing and local monotonicity: if $\Delta P_{\text{meas}}$ is outside $[\Delta P_{\text{VLP}}(q_{\min}), \Delta P_{\text{VLP}}(q_{\max})]$ or multiple roots exist or local sensitivity $\left|\frac{\partial \Delta P_{\text{VLP}}}{\partial q_{\text{liq}}}\right| < \epsilon_{\text{sens}}$ (e.g. $0.015\text{ psi/(STB/d)}$), return `np.nan` (`vlp_valid = False`) so WLS excludes VLP on ill-conditioned intervals.
- Also expose `invert_vlp_water_cut(...)` solving $\Delta P_{\text{VLP}}(q_{\text{liq}}, WC) = \Delta P_{\text{meas}}$ for $WC \in [0, 1]$ as an independent physical diagnostic.

---

### 3.4 (d) Inverse-Variance Weighted Least Squares (WLS) Data Reconciliation (`physics_vfm.py`)

Let $\mathcal{M} = \{\text{IPR}, \text{Choke}, \text{VLP}\}$ produce per-minute liquid-rate estimates $\hat{q}_{w,m}(t)$ whenever their required sensors are finite and validity checks pass.

1. **Per-well LOTO variance estimation (strictly causal in walk-forward):**
   From calibration residuals $e_{w,m,k} = \hat{q}_{w,m}(t_k) - q_{\text{liq},k}^{\text{test}}$ computed via **Leave-One-Test-Out** across the well's calibration tests ($\le t_{\max}$):
   $$\hat{\sigma}_{w,m}^2 = \max\!\left(\frac{1}{n_{\text{valid}}} \sum_{k} e_{w,m,k}^2, \; (0.015\,\bar{q}_w)^2, \; 15.0^2\right).$$
   - The relative floor $(1.5\% \cdot \bar{q}_w)^2$ prevents small-sample LOTO luck from giving one estimator near-infinite weight.
   - If an estimator has $< 2$ valid LOTO residuals (or failed calibration), set $w_{w,m} = 0$ (or assign high fallback variance).
2. **Per-minute WLS fusion (over valid subset $\mathcal{M}_t \subseteq \mathcal{M}$):**
   $$w_{w,m} = \frac{1}{\hat{\sigma}_{w,m}^2}, \qquad \hat{q}_{\text{liq}}^{\text{WLS}}(t) = \frac{\sum_{m \in \mathcal{M}_t} w_{w,m}\,\hat{q}_{w,m}(t)}{\sum_{m \in \mathcal{M}_t} w_{w,m}}, \qquad \sigma_{\text{WLS}}(t) = \left(\sum_{m \in \mathcal{M}_t} w_{w,m}\right)^{-1/2}.$$
   - If $\mathcal{M}_t = \emptyset$ (e.g., sensor outage on all estimators) and `not is_shut_in`, emit `np.nan` and set `is_sensor_missing = True` (never silently impute global training means).
   - **Consistency diagnostic:** $\chi^2(t) = \sum_{m \in \mathcal{M}_t} \frac{(\hat{q}_{w,m}(t) - \hat{q}_{\text{liq}}^{\text{WLS}}(t))^2}{\hat{\sigma}_{w,m}^2}$ and normalized discrepancy flags (`chi2 > 9.21` for $2$ dof indicates single-sensor drift or unmodeled transient).

---

### 3.5 (e) Slow Physical Trends $WC_w(t)$ and $GOR_w(t)$ (`physics_vfm.py`)

Both $WC$ and $GOR$ reflect reservoir fluid/saturation evolution, NOT instantaneous wellhead pressure fluctuations:
1. **Water Cut $WC_w(t)$:**
   Model per well as a smooth, bounded function of time $t_{\text{days}}$ (or cumulative liquid/oil production):
   - Ridge-regularized linear trend (or bounded logistic trend) fitted to causal well-test `water_cut_frac`:
     $$WC_w(t) = \text{clip}\!\left(\hat{a}_w + \hat{b}_w\,(t_{\text{days}} - \bar{t}_w),\; 0.0,\; 0.99\right),$$
     with slope shrinkage toward a weak non-negative prior (e.g. penalty $\lambda_b$ when $n_{\text{tests}} \le 3$, and clamping $\hat{b}_w \ge 0$ if waterflood monotonicity is enforced).
   - For walk-forward CV at test $k$, fit $WC_w(t)$ strictly using tests $1 \dots k-1$; when $k=1$, only prior test $1$ is available (zero slope fallback).
2. **Formation GOR $GOR_w(t)$:**
   Because $P_{wf} > P_b$ holds everywhere, producing formation GOR is anchored to $R_{sb,w} = \text{solution\_gor\_at\_pb\_scf\_stb}$ with empirical Bayes shrinkage of the causal well-test mean $\overline{GOR}_w$:
   $$\widehat{GOR}_w = \text{clip}\!\left(\frac{n_w\,\overline{GOR}_w + \kappa\,R_{sb,w}}{n_w + \kappa},\; 0.85\,R_{sb,w},\; 1.15\,R_{sb,w}\right), \qquad \kappa = 3.0.$$
   *(If $P_{wf} < P_b$ were ever encountered, scale $GOR$ smoothly above $R_{sb}$ as free gas evolves in the reservoir.)*

---

### 3.6 (f) Phase Reconstruction & Total Gas Balance

For all active (`not is_shut_in`) minutes with finite $\hat{q}_{\text{liq}}$:
$$q_o(t) = \hat{q}_{\text{liq}}(t)\,(1 - WC_w(t)), \qquad q_w(t) = \hat{q}_{\text{liq}}(t)\,WC_w(t),$$
$$q_{g,\text{form}}(t) = \frac{q_o(t) \cdot GOR_w(t)}{1000.0}, \qquad q_{g,\text{tot}}(t) = q_{g,\text{form}}(t) + \text{clip}(q_{\text{gl}}(t), 0, \infty).$$
For shut-in minutes (`choke_opening_pct <= 0.5`), set $q_{\text{liq}} = q_o = q_w = q_{g,\text{form}} = q_{g,\text{tot}} = 0.0$.
*(Note: When `choke_opening_pct > 0.5` and `dp_choke <= 2.0` such as `GL-03 2025-09-06 08:20`, choke yields `0.0` while `is_shut_in` for PBU purposes remains `False`; check `dp_choke <= 2.0` explicitly as a flow stall/transient flag rather than a PBU trigger.)*

---

### 3.7 (g) Daily Integration & Exact-Window Well-Test Anchoring (`07_physics_vfm_daily.py`)

1. **Calendar-complete daily grid:** Reindex all 10 wells over every calendar UTC date `[2025-01-01 .. 2025-09-30]` (`273` days $\times$ `10` wells = `2730` rows) so SCADA-outage days (`GL-06` `2025-06-24`) are never omitted.
2. **True uptime vs telemetry completeness:**
   - `n_expected_minutes = 1440`
   - `n_observed_minutes = count(timestamps on date)`
   - `n_valid_minutes = count(finite prediction or shut-in minutes on date)`
   - `flowing_minutes = count(~is_shut_in on date)`
   - `uptime_frac = flowing_minutes / max(n_observed_minutes, 1)` (effective operational uptime) and `data_completeness = n_valid_minutes / 1440.0`.
   - Daily unanchored rate = mean over valid minutes (shut-ins contribute `0.0`; `NaN` sensor minutes are excluded from mean, not filled with `0.0`, unless the entire day is missing `n_valid_minutes == 0`, in which case linearly interpolate from nearest valid days and set `is_imputed_day = True`).
3. **Exact-window test anchoring (fixes `test_start.floor("D")` midnight bug):**
   - For each well test $k$, compute the unanchored VFM reference rates $\bar{q}_{\text{VFM},k}$ over the **exact steady-flowing `[test_start, test_end]` interval** (matching `window_aligner.py`), NOT `test_start.floor("D")`.
   - Place each calibration anchor at the test midpoint $t_{\text{mid},k} = \frac{1}{2}(t_{\text{start},k} + t_{\text{end},k})$.
   - Compute phase multipliers $m_{\phi,k} = \text{clip}\!\left(q_{\phi,k}^{\text{test}} / \max(\bar{q}_{\text{VFM},\phi,k}, 1.0),\; 0.75,\; 1.35\right)$ for $\phi \in \{\text{oil}, \text{water}, \text{gas\_form}\}$ and interpolate piecewise-linearly at daily midpoints $t_d + 12\text{h}$ (flat clamp outside first/last test).
   - Apply $m_{\phi}(t_d)$ to daily VFM rates (shut-in `0.0` stays `0.0`), recompute `liquid_rate_alloc_stbd = oil_rate_alloc_stbd + water_rate_alloc_stbd`, and compute `total_gas_rate_alloc_mscfd = gas_rate_alloc_mscfd + daily_gas_lift_mscfd` (respecting shut-in zeroing).

---

## 4. Module Interface Contracts

### 4.1 `vfm.physics.reservoir`
- `compute_wellbore_fluid_gradient(api, wc_frac, gor_scf_stb=0.0, gas_sg=0.7, bo=1.14, bw=1.02, salinity_ppm=100000.0) -> float | np.ndarray`
- `gauge_to_datum_pressure(p_gauge_psia, gauge_tvd_ft, datum_tvd_ft, fluid_grad_psi_ft) -> float | np.ndarray`
- `ipr_rate(J, Pr, Pwf, Pb=None) -> float | np.ndarray`
- `estimate_pr_from_buildup(hf_well_df, shut_in_start, shut_in_end, prior_rate_stbd, pvt_row, comp_row, fluid_grad_psi_ft=None, wc_frac=0.0) -> dict`
- `fit_pr_and_pi(well_tests, hf_cache_by_well, pvt, completion, max_time=None) -> dict[str, dict]`
- `predict_ipr_q_liq(well_params, timestamps, dh_pressure_psia, wc_frac=None, oil_api=None, gas_sg=None, salinity_ppm=None) -> np.ndarray`

### 4.2 `vfm.physics.choke`
- `choke_bean_diameter_in(choke_opening_pct, choke_max_size_64ths=128.0) -> float | np.ndarray`
- `choke_rate(choke_opening_pct, whp_psia, whp_downstream_psia, wc_frac, gor_scf_stb, gas_lift_rate_mscfd, oil_api, gas_sg, cd=0.78, choke_max_size_64ths=128.0, water_salinity_ppm=100000.0, wh_temp_degf=120.0, r_crit=0.55) -> float | np.ndarray`
- `calibrate_choke(well_tests, hf_cache_by_well, completion, pvt=None, max_time=None) -> dict[str, dict]` (returns per-well `cd`, `r_crit`, `sigma_choke`, `loto_mae_stbd`, `loto_rmse_stbd`, static completion fields)
- `predict_choke_q_liq(well_params, choke_opening_pct, whp_psia, whp_downstream_psia, gas_lift_rate_mscfd, wc_frac, gor_scf_stb, wh_temp_degf=None) -> np.ndarray` (`0.0` when closed or $\Delta P \le 0$; `np.nan` when required sensors are `NaN`)

### 4.3 `vfm.physics.vlp`
- `vlp_pressure_drop(q_liq_stbd, wc_frac, whp_psia, gas_lift_rate_mscfd, gor_scf_stb, oil_api, gas_sg, lift_gas_sg, gauge_tvd_ft, gl_valve_tvd_ft, tubing_id_in, res_temp_degf, bo=1.14, bw=1.02, water_salinity_ppm=100000.0, wh_temp_degf=120.0, c_slip=1.2, f_mult=1.0) -> float`
- `invert_vlp_q_liq(dp_tubing_psia, wc_frac, whp_psia, gas_lift_rate_mscfd, ..., min_sens_psi_per_stbd=0.015) -> float` (returns `np.nan` if unbracketed or ill-conditioned)
- `invert_vlp_water_cut(dp_tubing_psia, q_liq_stbd, whp_psia, gas_lift_rate_mscfd, ...) -> float`
- `calibrate_vlp(well_tests, completion, pvt, max_time=None) -> dict[str, dict]` (returns per-well `c_slip`, `f_mult`, `sigma_vlp`, `loto_mae_stbd`, `loto_rmse_stbd`, `n_valid_loto`, static parameters)
- `predict_vlp_q_liq(well_params, whp_psia, dh_pressure_psia, gas_lift_rate_mscfd, wc_frac, gor_scf_stb) -> np.ndarray` (returns `np.nan` where sensors are `NaN` or inversion is ill-conditioned)

### 4.4 `vfm.modeling.physics_vfm.PhysicsVFM`
- `fit(well_tests_df, hf_cache_by_well, completion_df, pvt_df, max_time=None) -> PhysicsVFM`
  - Filters `well_tests_df` (`test_end <= max_time`) and `hf_cache_by_well` (`timestamp <= max_time`) when `max_time` is provided (zero walk-forward leakage!).
- `predict(hf_df) -> pd.DataFrame`
  - Returns columns: `liquid_rate_stbd`, `oil_rate_stbd`, `water_rate_stbd`, `gas_rate_mscfd`, `total_gas_rate_mscfd`, `water_cut_pct`, `gor_scf_stb`, `q_liq_ipr`, `q_liq_choke`, `q_liq_vlp`, `wls_sigma_stbd`, `reconcile_chi2`, `is_shut_in`, `is_transient`, `is_sensor_missing`.
