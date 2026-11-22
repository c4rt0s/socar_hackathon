# Front-end spec: tab_models.js (Model comparison + Correlations)

Write `dashboard/static/tab_models.js`: an IIFE that waits for `window.WB` (it is defined before this
script runs) and calls WB.registerTab twice. Use only WB helpers, the DOM and Chart.js (global `Chart`).
Every server string inserted as HTML must go through WB.esc. Inject its own CSS with a <style> element
(prefix classes with `tm-`), using the page's CSS variables when available (`--border`, `--muted`, `--bg`)
with fallbacks.

## Tab 4 — Model comparison (order 4, id "models")
Data: GET /api/models (models[] with name, family, description, causal, pooled, status; active;
benchmark_status), GET /api/benchmark (available, status, summary[], per_well[], points[]),
GET /api/models/well?name=W&models=a,b,c.
1. Header line: benchmark status; button "Re-run benchmark" → WB.post('/api/benchmark/run', {}) then poll
   /api/models every 5 s while benchmark_status starts with "running"; when it changes, reload
   /api/benchmark and re-render. Note: "Walk-forward is strictly causal; LOTO leaves one test out;
   test_linear is hindcast-only (LOTO only)".
2. Protocol toggle (walk_forward | loto) and metric select (liquid, oil, water, gas × MAE | MAPE).
3. Summary table from summary[] filtered by cv_mode: model, family (from /api/models), n_tests, for each
   phase MAE and MAPE and bias (`<phase>_rate_stbd_mae|mape|bias`, gas is `gas_rate_mscfd_*`). Sortable
   by clicking headers; best value per column bold; rows clickable (select model).
4. Horizontal bar chart of the selected metric per model (sorted ascending).
5. Per-well heat table: rows = models, columns = wells, cell = per_well.liquid_mape (cv_mode filter),
   background colour from green (low) to red (high) scaled to the table's min..max; tooltip with value.
6. Scatter chart true vs predicted liquid (points[] filtered by cv_mode and the selected model) with
   a y=x line; colour points by well (WB.palette).
7. Card "Daily comparison for <current well>": checkboxes for every model (default: physics_wls and
   the active model). Button "Compute selected" → WB.post('/api/models/compute', {names}) then poll
   /api/models until all selected are ready or error. Then GET /api/models/well?name=W&models=... and
   plot a line chart: per model `<phase>_vfm` (unanchored, toggle) or `<phase>_alloc`, with well tests
   as points (map t_mid to the `dates` label `YYYY-MM-DD`). Phase select oil/water/liquid/gas. Also a
   small table: model, cum_oil_alloc_stb, difference vs physics_wls in %.
   Re-render on WB.on('well').

## Tab 5 — Correlations (order 5, id "corr")
Data: GET /api/corr?source=S&well=W&method=M → {variables[], matrix[][], n_rows, wells[]};
GET /api/scatter?source=S&well=W&x=X&y=Y → {points:[{well_name,x,y}], pearson_r, n}.
1. Controls: source (well_tests | daily | hourly) with a one-line explanation of each
   ("well tests: 85 measured tests", "daily: active allocation model + daily sensor means",
   "hourly: hourly sensor means"), well (all + overview wells), method (pearson | spearman | kendall),
   a text filter to show only variables whose name contains a substring, and "|r| ≥" threshold slider
   (0–0.9) that dims weaker cells.
2. Heatmap as an HTML table: header row with variable names rotated -60°, first column names; cells
   coloured with a diverging scale (−1 blue #2166ac → 0 white → +1 red #b2182b), the value with 2 dp
   inside when the matrix has ≤ 20 variables, title attribute "x vs y: r". Null → grey. Clicking a cell
   selects that pair.
3. Scatter for the selected pair (Chart.js scatter, colour by well with WB.palette, legend by well),
   header with r and n.
4. "Strongest pairs" list: top 15 |r| off-diagonal unique pairs (click to select).
Reload on control changes; when the well changes in the sidebar and the well select is not "all",
follow the sidebar well.
