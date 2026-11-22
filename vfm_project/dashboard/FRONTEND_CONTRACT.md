# Shared front-end contract (index.html <-> tab scripts)

Files: `dashboard/static/index.html` (shell + core tabs) and `dashboard/static/tab_models.js`
(extra tabs). Plain browser JS, no build step, no modules. Chart.js 4.4.1 is loaded by index.html from
`https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js` (no other external resources).
index.html loads `/static/tab_models.js` with a normal `<script defer>` AFTER defining `window.WB`.

`window.WB` (defined in index.html):

```
WB.api(path)                    -> Promise<json>  GET; rejects Error(message from {"error"} or HTTP text)
WB.post(path, body, busyMsg)    -> Promise<json>  POST JSON; shows the busy overlay while pending,
                                                    shows an error toast and rethrows on failure
WB.fmt(v, digits=1)             -> string, '—' for null/undefined/NaN; digits applied with toFixed
WB.esc(s)                       -> HTML-escaped string (use for every server string put in innerHTML)
WB.toast(msg, level='info')     -> level in info|warn|error|success
WB.busy(on, msg)
WB.state                        -> { overview, currentWell, currentEvent, activeModel }
WB.on(evt, fn) / WB.emit(evt, data)   events: 'well' (well name), 'overview' (overview json),
                                        'models' (models json), 'tab' (tab id)
WB.registerTab({id, title, order, render(container), onShow(), onWellChange(well)})
    render(container) is called once with an empty <div>; onShow() each time the tab is opened;
    onWellChange(well) when the sidebar selection changes (only if the tab has been rendered).
WB.palette                      -> array of 10 distinct colours (index by well or model)
WB.phaseUnit(phase)             -> 'STB/d' or 'Mscf/d' (gas)
WB.chart(canvas, config)        -> creates a Chart.js chart, destroying any previous chart on that canvas
```

Tabs (order): 1 Allocation, 2 PTA, 3 PTA ↔ VFM link (index.html); 4 Model comparison, 5 Correlations
(tab_models.js). The left sidebar (well list) and header are shared. Selecting a well emits 'well'.
All numbers from the API may be null. Long POSTs (5–120 s) must use WB.post so the busy overlay shows.
Models that are "running" must be polled with GET /api/models every 3 s until their status starts with
"ready" or "error".
