# Front-end spec v2 (tabs): VFM + PTA engineering workbench

Users: petroleum engineers. They compare allocation (rate) models, review pressure build-up tests
(PTA), decide which PTA results may influence the VFM (baseline = uncoupled), and explore correlations.
Clean light theme, CSS variables, system font, cards with subtle borders, sidebar ~280 px, works at
1280 px. Charts ~260 px high, `animation:false`. Every server string inserted as HTML goes through WB.esc.
Numbers: rates 0 dp, pressures 1 dp, skin 2 dp, kh 0 dp, PI 3 dp, ratios 3 dp, MAPE 2 dp + '%'.

## Shell (index.html)
- Startup: poll GET /api/status every 2 s until ready (show status text, error in red), then
  GET /api/overview and GET /api/models, select the first well, open the Allocation tab.
- Header: title "VFM + PTA Workbench", config path (small), switch "PTA → VFM coupling" bound to
  overview.settings.enabled (POST /api/settings {"enabled": bool}) with badge "BASELINE (coupling off)"
  grey / "COUPLED" amber, badge "Active model: <name>", buttons "Export scenario" (prompt name,
  POST /api/export, toast the folder) and "Reset judgement" (confirm, POST /api/reset).
- Sidebar: wells table (name, n_tests, one chip per PTA event coloured RADIAL green / NO_RADIAL or
  RADIAL_FORCED orange / FAILED red, cum_oil_change_pct signed 2 dp ('—' if null/0), dot if coupled);
  click selects the well (WB.emit('well')). Below: overview.notes coloured INFO/WARN/BLOCK.
- Tab bar from WB.registerTab registrations, sorted by order.

## Tab 1 — Allocation
Uses GET /api/models (list, statuses, active), GET /api/models/well?name=W&models=a,b,
GET /api/well?name=W.
- Model picker: a select of all models (show family and description as option title; mark
  "(hindcast)" when causal=false). Changing it POSTs /api/models/active {"name"}; poll /api/models
  until that model is ready, then redraw. Show the status next to the picker.
- Phase buttons: oil, water, liquid, gas.
- Chart "Allocated rates": lines physics baseline (`models_well.models.physics_wls[phase+'_alloc']`),
  active model allocated (`[phase+'_alloc']`) and active model unanchored (`[phase+'_vfm']`, dashed),
  coupled physics (`well.coupled[phase]`) only when coupling is enabled, plus well-test points
  (`tests[].t_mid` → `YYYY-MM-DD` label of `dates`). Category x-axis = `dates`.
- Chart "Physics estimators": well.coupled.q_ipr, q_choke, q_vlp, liquid_vfm and dashed
  well.baseline.q_ipr. Estimator checkboxes bound to well.estimators → POST /api/settings
  {"estimators": {W: {ipr, choke, vlp}}}.
- Chart "Reservoir pressure & well condition": well.baseline.pr_datum and well.coupled.pr_datum
  (left axis, psia), well.coupled.j_multiplier (right axis, stepped); anchors from
  well.ipr.anchors_baseline / anchors_coupled as points. Table well.ipr.baseline vs coupled
  (J, Pr0_datum_psia, decline_psi_per_day, loto_mae_stbd, pr_prior_source).
- Card "Well-condition changes": table of well.property_changes (enabled checkbox, delete) + add form
  (date, skin, kh_md_ft, note). Save posts the FULL list {"property_changes": [...]}, built from
  overview.settings.property_changes with this well's rows replaced. Show well.condition_timeline and
  well.notes.

## Tab 2 — PTA
GET /api/pta?event=ID for events of the current well (from well.events or overview.events).
- Event list (id, start, duration, status, kh, skin). Selecting loads the event.
- Log-log chart (scatter, log x and y): Δp (series.dp_psi) and derivative (series.derivative_psi,
  skip nulls) vs series.dt_hr; derivative points inside radial_interval.dt_start_hr..dt_end_hr as a
  separate highlighted dataset; model-match curves methods.model_match.dt_model_hr vs dp_model_psi and
  deriv_model_psi as lines (if present).
- Card "Methods": table rows Derivative (results: kh_md_ft, k_md, apparent_skin,
  extrapolated_pressure_psi, storage_bbl_per_psi), Horner (methods.horner: kh, k, skin, p_star, m, r2),
  MDH (methods.mdh: kh, k, skin, valid), Model match (methods.model_match: kh_md_ft, k_md, skin,
  c_bbl_psi, rms_psi, converged), Internal (internal_pbu: kh_md_ft, skin, pr_datum_psia,
  quality_flag). Show methods.window_source and all `notes` arrays as small text.
- Card "Productivity & pressure": methods.productivity (pi_actual_stbd_psi, pi_theoretical_stbd_psi,
  pi_ideal_stbd_psi, flow_efficiency, damage_ratio, productivity_ratio_actual_to_theoretical,
  p_avg_source), derived (dietz_pbar_gauge_psia, dietz_pbar_datum_psia, p_last_datum_psia,
  dietz_note, dietz_tpda), results.investigation_radius_ft, rate_note, rate_history.
- Card "QC flags" coloured by severity.
- Card "Re-run analysis" bound to `settings`: rate_source (vfm_daily / well_tests),
  prepend_cumulative_history, smoothing_L10, manual radial start/end (empty = automatic),
  force_manual_radial checkbox ("engineer override, recorded as WARN"). POST /api/pta/rerun
  {"event", "overrides": {..., "manual_radial_dt_hr": [s, e] or null}} → re-render with the response.
- Decision editing lives in Tab 3 (link a button "Edit decision" that switches to Tab 3 and focuses it).

## Tab 3 — PTA ↔ VFM link
GET /api/link (rows for all wells, settings, notes, enabled).
- Explain in one line: "Baseline VFM ignores PTA unless coupling is on and a build-up is set to Use".
- Comparison table, one row per event, grouped column headers:
  kh (derivative | Horner | model | from VFM J), skin (derivative | Horner | model),
  pressure (p* gauge | p̄ datum PTA | p_last datum | Pr datum VFM baseline | Pr datum VFM coupled),
  productivity (PI actual | PI theoretical | PI ideal | J VFM baseline | J VFM coupled | PI/J ratio),
  flow efficiency, damage ratio. Highlight cells where PTA and VFM differ by more than 10 % (kh_derivative
  vs kh_from_vfm_j, pbar_datum_pta vs pr_datum_vfm_baseline, pi_actual_pta vs j_vfm_baseline).
- Decision editor for the selected row (row click), bound to settings.decisions[event_id]:
  use; apply_pressure + pressure_choice (dietz | p_last | manual) + manual_pr_datum_psia;
  apply_skin + skin_override; apply_kh + kh_override; apply_pi + pi_choice (actual | theoretical | manual)
  + pi_override; note. "Apply decision" → POST /api/settings {"decisions": {id: {...}}}; then refresh
  /api/link, /api/overview and the current well.
- Notes list (settings notes) coloured by level.
