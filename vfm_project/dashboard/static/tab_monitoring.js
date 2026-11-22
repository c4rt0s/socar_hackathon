(function () {
  'use strict';

  function boot() {
    if (!window.WB || typeof window.WB.registerTab !== 'function' || !window.AxisZoom) {
      setTimeout(boot, 25);
      return;
    }

    var WB = window.WB;
    var STYLE_ID = 'quiver-mon-styles';
    if (!document.getElementById(STYLE_ID)) {
      var st = document.createElement('style');
      st.id = STYLE_ID;
      st.textContent = [
        '.qv-root { display:flex; flex-direction:column; gap:14px; font-family:-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif; color:#182026; }',
        '.qv-topbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; background:#ffffff; border:1px solid #d8e1e8; border-radius:6px; padding:10px 14px; box-shadow:0 1px 2px rgba(16,22,26,0.04); }',
        '.qv-bar-left, .qv-bar-right { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }',
        '.qv-well-tag { font-weight:700; font-size:13px; color:#10161a; background:#ebf1f5; padding:4px 10px; border-radius:4px; letter-spacing:0.02em; }',
        '.qv-seg { display:inline-flex; border:1px solid #ced9e0; border-radius:4px; overflow:hidden; background:#f5f8fa; }',
        '.qv-seg button { border:none; background:transparent; padding:5px 11px; font-size:12px; font-weight:600; color:#394b59; cursor:pointer; transition:background 0.12s; }',
        '.qv-seg button + button { border-left:1px solid #d8e1e8; }',
        '.qv-seg button:hover { background:#ebf1f5; }',
        '.qv-seg button.active { background:#2d72d2; color:#ffffff; }',
        '.qv-select { border:1px solid #ced9e0; border-radius:4px; background:#ffffff; padding:5px 9px; font-size:12px; color:#182026; min-width:190px; }',
        '.qv-btn { border:1px solid #ced9e0; border-radius:4px; background:#ffffff; padding:5px 10px; font-size:12px; font-weight:600; color:#394b59; cursor:pointer; }',
        '.qv-btn:hover { background:#f5f8fa; border-color:#bfccd6; }',
        '.qv-check { display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:500; color:#394b59; cursor:pointer; user-select:none; }',
        '.qv-meta-pill { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; color:#5c7080; background:#f5f8fa; border:1px solid #e1e8ed; padding:3px 8px; border-radius:4px; }',
        '.qv-meta-pill.loading { background:#fff8e6; border-color:#f2b824; color:#a66321; }',
        '.qv-model-badge { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; font-weight:700; color:#0f9960; background:#e6f7f0; border:1px solid #a7e3c9; padding:2px 8px; border-radius:4px; }',
        '.qv-model-badge.computing { color:#bf7326; background:#fff6e6; border-color:#f0d19e; }',
        '.qv-kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; }',
        '@media (max-width: 980px) { .qv-kpi-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }',
        '.qv-kpi { background:#ffffff; border:1px solid #d8e1e8; border-radius:6px; padding:12px 14px; display:flex; flex-direction:column; gap:4px; box-shadow:0 1px 2px rgba(16,22,26,0.03); }',
        '.qv-kpi-lbl { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#5c7080; }',
        '.qv-kpi-val { font-size:21px; font-weight:700; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; color:#10161a; font-variant-numeric:tabular-nums; line-height:1.2; }',
        '.qv-kpi-sub { font-size:11.5px; color:#5c7080; font-variant-numeric:tabular-nums; }',
        '.qv-card { background:#ffffff; border:1px solid #d8e1e8; border-radius:6px; padding:14px 16px 12px; box-shadow:0 1px 2px rgba(16,22,26,0.04); }',
        '.qv-card-hdr { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; margin-bottom:10px; }',
        '.qv-card-title { font-size:13px; font-weight:700; color:#10161a; letter-spacing:0.01em; display:flex; align-items:center; gap:8px; }',
        '.qv-pills { display:flex; flex-wrap:wrap; gap:6px; }',
        '.qv-pill { display:inline-flex; align-items:center; gap:6px; padding:4px 9px; border-radius:4px; border:1px solid #d8e1e8; background:#f5f8fa; font-size:11.5px; font-weight:600; color:#5c7080; cursor:pointer; user-select:none; transition:all 0.12s; }',
        '.qv-pill:hover { background:#ebf1f5; }',
        '.qv-pill.on { background:#ffffff; color:#10161a; border-color:#8a9ba8; box-shadow:0 1px 2px rgba(16,22,26,0.06); }',
        '.qv-dot { width:8px; height:8px; border-radius:2px; display:inline-block; }',
        '.qv-canvas-wrap { position:relative; width:100%; user-select:none; }',
        '.qv-scrubber-card { background:#ffffff; border:1px solid #d8e1e8; border-radius:6px; padding:10px 14px; display:flex; flex-direction:column; gap:6px; }',
        '.qv-scrubber-wrap { position:relative; width:100%; height:48px; background:#f5f8fa; border:1px solid #e1e8ed; border-radius:4px; overflow:hidden; cursor:grab; user-select:none; }',
        '.qv-scrubber-foot { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; font-size:11px; color:#5c7080; }'
      ].join('\n');
      document.head.appendChild(st);
    }

    // Only output channels for the currently selected VFM method
    var PROD_SERIES = [
      { key: 'oil_rate_vfm_stbd', label: 'Oil Rate', unit: 'STB/d', color: '#0f9960', dash: [], width: 2.1, defaultOn: true },
      { key: 'water_rate_vfm_stbd', label: 'Water Rate', unit: 'STB/d', color: '#0284c7', dash: [], width: 1.8, defaultOn: false },
      { key: 'gas_rate_vfm_mscfd', label: 'Formation Gas', unit: 'Mscf/d', color: '#db3737', dash: [], width: 1.8, defaultOn: false },
      { key: 'liquid_rate_vfm_stbd', label: 'Total Liquid', unit: 'STB/d', color: '#634dbf', dash: [4, 2], width: 1.7, defaultOn: false }
    ];

    var TELEM_SERIES = [
      { key: 'dh_pressure_psia', label: 'PDG Pressure', unit: 'psia', color: '#184a90', axis: 'left', step: false, defaultOn: true },
      { key: 'whp_psia', label: 'WHP', unit: 'psia', color: '#137cbd', axis: 'left', step: false, defaultOn: true },
      { key: 'choke_opening_pct', label: 'Choke %', unit: '%', color: '#d9822b', axis: 'right', step: true, defaultOn: true },
      { key: 'whp_downstream_psia', label: 'Downstream P', unit: 'psia', color: '#7157d9', axis: 'left', step: false, defaultOn: false },
      { key: 'annulus_pressure_psia', label: 'Annulus P', unit: 'psia', color: '#5c7080', axis: 'left', step: false, defaultOn: false },
      { key: 'gas_lift_rate_mscfd', label: 'Gas Lift', unit: 'Mscf/d', color: '#9b2c9b', axis: 'right', step: false, defaultOn: false },
      { key: 'dh_temperature_degf', label: 'Temp °F', unit: '°F', color: '#c23030', axis: 'right', step: false, defaultOn: false }
    ];

    var root = null;
    var payload = null;
    var prodChart = null;
    var telemChart = null;
    var prodZoomHandle = null;
    var telemZoomHandle = null;
    var loadSeq = 0;

    var state = {
      rangeKey: 'all',
      plotMode: 'box', // 'box' (drag zoom region) | 'pan' (drag shift X & Y)
      viewStartMs: null,
      viewEndMs: null,
      showShutins: true,
      showTests: true,
      prodOn: {},
      telemOn: {},
      hoverMs: null,
      yBounds: { prod: null, telemLeft: null, telemRight: null }
    };
    var fetchTimer = null;
    PROD_SERIES.forEach(function (s) { state.prodOn[s.key] = s.defaultOn; });
    TELEM_SERIES.forEach(function (s) { state.telemOn[s.key] = s.defaultOn; });

    var brushDrag = null;

    function pad2(n) { return n < 10 ? '0' + n : String(n); }
    function fmtTime(ms, spanMs) {
      if (!isFinite(ms)) return '';
      var d = new Date(ms);
      var Y = d.getUTCFullYear(), M = pad2(d.getUTCMonth() + 1), D = pad2(d.getUTCDate());
      var h = pad2(d.getUTCHours()), m = pad2(d.getUTCMinutes());
      if (spanMs > 25 * 86400000) return Y + '-' + M + '-' + D;
      return M + '-' + D + ' ' + h + ':' + m;
    }
    function fmtNum(v, d) {
      if (v === null || v === undefined || !isFinite(v)) return '—';
      return Number(v).toFixed(d === undefined ? 1 : d);
    }
    function arrayStats(arr) {
      if (!arr || !arr.length) return { mean: null, min: null, max: null, n: 0 };
      var sum = 0, n = 0, mn = Infinity, mx = -Infinity;
      for (var i = 0; i < arr.length; i++) {
        var v = arr[i];
        if (v !== null && v !== undefined && isFinite(v)) {
          sum += v; n++;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
      return n ? { mean: sum / n, min: mn, max: mx, n: n } : { mean: null, min: null, max: null, n: 0 };
    }

    function setLoading(on) {
      if (!root) return;
      var el = root.querySelector('#qv-status');
      if (!el) return;
      if (on) {
        el.classList.add('loading');
        el.textContent = 'Querying telemetry...';
      } else {
        el.classList.remove('loading');
        if (payload) {
          el.textContent = payload.actual_res + ' · ' + payload.n_points.toLocaleString() + ' pts';
        }
      }
    }

    function syncXView(range, triggerFetch, sourceChart) {
      if (!payload || !payload.bounds_ms || !range) return;
      var b = payload.bounds_ms;
      var newMin = Math.max(b[0], Math.round(range[0]));
      var newMax = Math.min(b[1], Math.round(range[1]));
      if (newMax - newMin < 600000) return;

      var isAll = (newMin <= b[0] + 60000 && newMax >= b[1] - 60000);
      state.rangeKey = isAll ? 'all' : 'custom';
      state.viewStartMs = isAll ? null : newMin;
      state.viewEndMs = isAll ? null : newMax;
      updateRangeButtons();

      [prodChart, telemChart].forEach(function (c) {
        if (c && c !== sourceChart && c.scales && c.scales.x) {
          c.options.scales.x.min = newMin;
          c.options.scales.x.max = newMax;
          c.scales.x.options.min = newMin;
          c.scales.x.options.max = newMax;
          c.update('none');
        }
      });
      payload.view_ms = [newMin, newMax];
      drawScrubber();

      if (triggerFetch) {
        if (fetchTimer) clearTimeout(fetchTimer);
        fetchTimer = setTimeout(function () { fetchTelemetry(); }, 220);
      }
    }

    function makeOverlayPlugin(chartId) {
      return {
        id: 'qvOverlay_' + chartId,
        beforeDatasetsDraw: function (chart) {
          if (!payload) return;
          var ctx = chart.ctx, area = chart.chartArea, xScale = chart.scales.x;
          if (!area || !xScale) return;
          ctx.save();
          ctx.beginPath();
          ctx.rect(area.left, area.top, area.right - area.left, area.bottom - area.top);
          ctx.clip();

          if (state.showShutins && payload.shut_ins) {
            payload.shut_ins.forEach(function (si) {
              var x0 = xScale.getPixelForValue(si.start_ms);
              var x1 = xScale.getPixelForValue(si.end_ms);
              if (x1 < area.left || x0 > area.right) return;
              var l = Math.max(area.left, x0), r = Math.min(area.right, x1);
              ctx.fillStyle = 'rgba(219, 55, 55, 0.11)';
              ctx.fillRect(l, area.top, Math.max(2, r - l), area.bottom - area.top);
              ctx.strokeStyle = 'rgba(219, 55, 55, 0.45)';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(l, area.top); ctx.lineTo(l, area.bottom);
              ctx.stroke();
            });
          }

          if (state.showTests && payload.tests) {
            ctx.strokeStyle = 'rgba(15, 153, 96, 0.28)';
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1;
            payload.tests.forEach(function (t) {
              var xt = xScale.getPixelForValue(t.t_mid_ms);
              if (xt < area.left || xt > area.right) return;
              ctx.beginPath();
              ctx.moveTo(xt, area.top); ctx.lineTo(xt, area.bottom);
              ctx.stroke();
            });
            ctx.setLineDash([]);
          }
          ctx.restore();
        },
        afterDatasetsDraw: function (chart) {
          var ctx = chart.ctx, area = chart.chartArea, xScale = chart.scales.x;
          if (!area || !xScale) return;
          ctx.save();
          if (state.hoverMs !== null && isFinite(state.hoverMs)) {
            var xh = xScale.getPixelForValue(state.hoverMs);
            if (xh >= area.left && xh <= area.right) {
              ctx.strokeStyle = 'rgba(45, 114, 210, 0.55)';
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 3]);
              ctx.beginPath();
              ctx.moveTo(xh, area.top); ctx.lineTo(xh, area.bottom);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
          ctx.restore();
        }
      };
    }

    function wireCanvas(canvas, getChart, chartId) {
      var handle = window.AxisZoom.attach(canvas, getChart, {
        xScaleId: 'x',
        yLeftScaleId: chartId === 'prod' ? 'y' : 'left',
        rightScaleId: chartId === 'telem' ? 'right' : null,
        wheelPlotMode: 'x',
        boxZoomMode: 'x',
        getPlotMode: function () { return state.plotMode; },
        getXLimits: function () {
          return (payload && payload.bounds_ms)
            ? { minVal: payload.bounds_ms[0], maxVal: payload.bounds_ms[1], minSpan: 600000 }
            : null;
        },
        getYLeftLimits: function () {
          return null;
        },
        onAxisChange: function (scaleId, range) {
          if (scaleId === 'x') {
            syncXView(range, false, getChart());
          } else if (chartId === 'prod' && scaleId === 'y') {
            state.yBounds.prod = range;
          } else if (chartId === 'telem' && scaleId === 'left') {
            state.yBounds.telemLeft = range;
          } else if (chartId === 'telem' && scaleId === 'right') {
            state.yBounds.telemRight = range;
          }
        },
        onXViewCommit: function (range) {
          syncXView(range, true, null);
        },
        onReset: function (zone) {
          if (zone === 'y-left') {
            if (chartId === 'prod') state.yBounds.prod = null;
            else state.yBounds.telemLeft = null;
            renderCharts();
          } else if (zone === 'y-right' && chartId === 'telem') {
            state.yBounds.telemRight = null;
            renderCharts();
          } else {
            state.yBounds = { prod: null, telemLeft: null, telemRight: null };
            applyRangePreset('all');
          }
        }
      });

      canvas.addEventListener('mousemove', function (e) {
        if ((prodZoomHandle && prodZoomHandle.ui.drag) || (telemZoomHandle && telemZoomHandle.ui.drag)) return;
        var ch = getChart();
        if (!ch || !ch.scales.x || !ch.chartArea) return;
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        if (x >= ch.chartArea.left && x <= ch.chartArea.right) {
          state.hoverMs = ch.scales.x.getValueForPixel(x);
          if (prodChart) prodChart.draw();
          if (telemChart) telemChart.draw();
        }
      });
      canvas.addEventListener('mouseleave', function () {
        if ((prodZoomHandle && prodZoomHandle.ui.drag) || (telemZoomHandle && telemZoomHandle.ui.drag)) return;
        state.hoverMs = null;
        if (prodChart) prodChart.draw();
        if (telemChart) telemChart.draw();
      });
      return handle;
    }

    function buildSeriesXY(tMs, vals) {
      var pts = new Array(tMs.length);
      for (var i = 0; i < tMs.length; i++) {
        pts[i] = { x: tMs[i], y: vals && vals[i] !== undefined ? vals[i] : null };
      }
      return pts;
    }

    function renderKPIs() {
      if (!root || !payload) return;
      var s = payload.series || {};
      var oilV = arrayStats(s.oil_rate_vfm_stbd);
      var watV = arrayStats(s.water_rate_vfm_stbd);
      var gasV = arrayStats(s.gas_rate_vfm_mscfd);
      var dhp = arrayStats(s.dh_pressure_psia);
      var whp = arrayStats(s.whp_psia);
      var chk = arrayStats(s.choke_opening_pct);
      var gl = arrayStats(s.gas_lift_rate_mscfd);
      var activeM = payload.vfm_model || WB.state.activeModel || 'synth_stacked_v2';

      var badge = root.querySelector('#qv-vfm-badge');
      if (badge) {
        if (payload.vfm_model_ready === false) {
          badge.className = 'qv-model-badge computing';
          badge.textContent = 'Computing ' + (payload.vfm_requested_model || activeM) + '… (showing fallback)';
        } else {
          badge.className = 'qv-model-badge';
          badge.textContent = 'Model: ' + activeM;
        }
      }

      var chkArr = s.choke_opening_pct || [];
      var flowCnt = 0, totCnt = 0;
      for (var i = 0; i < chkArr.length; i++) {
        if (chkArr[i] !== null && isFinite(chkArr[i])) {
          totCnt++;
          if (chkArr[i] > 0.5) flowCnt++;
        }
      }
      var upPct = totCnt ? (100 * flowCnt / totCnt) : 100;
      var spanDays = payload.view_ms ? ((payload.view_ms[1] - payload.view_ms[0]) / 86400000) : 0;

      root.querySelector('#qv-kpi-oil').textContent = fmtNum(oilV.mean, 1) + ' STB/d';
      root.querySelector('#qv-kpi-oil-sub').textContent =
        activeM + ' · Water ' + fmtNum(watV.mean, 0) + ' · Gas ' + fmtNum(gasV.mean, 0) + ' Mscf/d';

      root.querySelector('#qv-kpi-dhp').textContent = fmtNum(dhp.mean, 1) + ' psia';
      root.querySelector('#qv-kpi-dhp-sub').textContent =
        'Range: ' + fmtNum(dhp.min, 0) + ' – ' + fmtNum(dhp.max, 0) + ' psia';

      root.querySelector('#qv-kpi-whp').textContent = fmtNum(whp.mean, 1) + ' psia';
      root.querySelector('#qv-kpi-whp-sub').textContent =
        'Choke ' + fmtNum(chk.mean, 1) + '% · Gas Lift ' + fmtNum(gl.mean, 0) + ' Mscf/d';

      root.querySelector('#qv-kpi-up').textContent = fmtNum(upPct, 1) + '%';
      root.querySelector('#qv-kpi-up-sub').textContent =
        fmtNum(spanDays, 1) + ' days · ' + (payload.shut_ins ? payload.shut_ins.length : 0) + ' PBUs · ' + (payload.tests ? payload.tests.length : 0) + ' tests';
    }

    function renderCharts() {
      if (!root || !payload) return;
      var tMs = payload.t_ms || [];
      var view = payload.view_ms || payload.bounds_ms;
      var spanMs = Math.max(1, view[1] - view[0]);
      var ser = payload.series || {};
      var activeM = payload.vfm_model || WB.state.activeModel || 'synth_stacked_v2';

      // 1. Production & Active VFM Chart (ONLY the selected VFM method!)
      var prodDatasets = [];
      PROD_SERIES.forEach(function (def) {
        if (!state.prodOn[def.key] || !ser[def.key]) return;
        prodDatasets.push({
          label: def.label + ' [' + activeM + '] (' + def.unit + ')',
          data: buildSeriesXY(tMs, ser[def.key]),
          borderColor: def.color,
          backgroundColor: def.color,
          borderDash: def.dash,
          borderWidth: def.width,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: true,
          tension: 0
        });
      });

      if (state.showTests && payload.tests && payload.tests.length) {
        if (state.prodOn.oil_rate_vfm_stbd) {
          prodDatasets.push({
            label: 'Separator Well Tests (Oil)',
            data: payload.tests.map(function (t) { return { x: t.t_mid_ms, y: t.oil_rate_vfm_stbd }; }),
            showLine: false,
            borderColor: '#10161a',
            backgroundColor: '#f5b041',
            borderWidth: 1.4,
            pointStyle: 'rectRot',
            pointRadius: 6,
            pointHoverRadius: 8
          });
        } else if (state.prodOn.water_rate_vfm_stbd) {
          prodDatasets.push({
            label: 'Separator Well Tests (Water)',
            data: payload.tests.map(function (t) { return { x: t.t_mid_ms, y: t.water_rate_vfm_stbd }; }),
            showLine: false,
            borderColor: '#10161a',
            backgroundColor: '#38bdf8',
            borderWidth: 1.4,
            pointStyle: 'rectRot',
            pointRadius: 6
          });
        } else if (state.prodOn.gas_rate_vfm_mscfd) {
          prodDatasets.push({
            label: 'Separator Well Tests (Gas)',
            data: payload.tests.map(function (t) { return { x: t.t_mid_ms, y: t.gas_rate_vfm_mscfd }; }),
            showLine: false,
            borderColor: '#10161a',
            backgroundColor: '#f87171',
            borderWidth: 1.4,
            pointStyle: 'rectRot',
            pointRadius: 6
          });
        } else if (state.prodOn.liquid_rate_vfm_stbd) {
          prodDatasets.push({
            label: 'Separator Well Tests (Liquid)',
            data: payload.tests.map(function (t) { return { x: t.t_mid_ms, y: t.liquid_rate_vfm_stbd }; }),
            showLine: false,
            borderColor: '#10161a',
            backgroundColor: '#a78bfa',
            borderWidth: 1.4,
            pointStyle: 'rectRot',
            pointRadius: 6
          });
        }
      }

      var prodCanvas = root.querySelector('#qv-prod-canvas');
      if (prodChart) prodChart.destroy();
      var prodPlugins = [makeOverlayPlugin('prod')];
      if (prodZoomHandle) prodPlugins.push(prodZoomHandle.hoverPlugin);
      prodChart = new Chart(prodCanvas.getContext('2d'), {
        type: 'line',
        data: { datasets: prodDatasets },
        plugins: prodPlugins,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          plugins: {
            legend: { display: true, position: 'top', align: 'end', labels: { boxWidth: 12, usePointStyle: true, font: { size: 11 } } },
            tooltip: {
              backgroundColor: 'rgba(16, 22, 26, 0.92)',
              titleFont: { family: 'ui-monospace, monospace', size: 11 },
              bodyFont: { family: 'ui-monospace, monospace', size: 11.5 },
              callbacks: {
                title: function (items) { return items.length ? fmtTime(items[0].parsed.x, 0) + ' UTC' : ''; }
              }
            }
          },
          scales: {
            x: {
              type: 'linear',
              min: view[0],
              max: view[1],
              grid: { color: '#edf2f7' },
              ticks: {
                maxTicksLimit: 9,
                font: { size: 11 },
                color: '#5c7080',
                callback: function (v) { return fmtTime(v, spanMs); }
              }
            },
            y: {
              beginAtZero: !state.yBounds.prod,
              min: state.yBounds.prod ? state.yBounds.prod[0] : undefined,
              max: state.yBounds.prod ? state.yBounds.prod[1] : undefined,
              afterFit: function (s) { s.width = Math.max(s.width, 64); },
              grid: { color: '#edf2f7' },
              ticks: { includeBounds: !state.yBounds.prod, font: { size: 11 }, color: '#5c7080' },
              title: { display: true, text: 'Rate (STB/d · Mscf/d)', font: { size: 11, weight: '600' }, color: '#5c7080' }
            }
          }
        }
      });

      // 2. Subsurface & Surface Telemetry Chart
      var telemDatasets = [];
      var hasRight = false;
      var rightUnits = [];
      TELEM_SERIES.forEach(function (def) {
        if (!state.telemOn[def.key] || !ser[def.key]) return;
        if (def.axis === 'right') {
          hasRight = true;
          if (rightUnits.indexOf(def.unit) < 0) rightUnits.push(def.unit);
        }
        telemDatasets.push({
          label: def.label + ' (' + def.unit + ')',
          data: buildSeriesXY(tMs, ser[def.key]),
          yAxisID: def.axis,
          borderColor: def.color,
          backgroundColor: def.color,
          borderWidth: def.step ? 1.5 : 1.7,
          stepped: def.step ? 'before' : false,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: true,
          tension: 0
        });
      });

      var telemCanvas = root.querySelector('#qv-telem-canvas');
      if (telemChart) telemChart.destroy();
      var telemPlugins = [makeOverlayPlugin('telem')];
      if (telemZoomHandle) telemPlugins.push(telemZoomHandle.hoverPlugin);
      telemChart = new Chart(telemCanvas.getContext('2d'), {
        type: 'line',
        data: { datasets: telemDatasets },
        plugins: telemPlugins,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          plugins: {
            legend: { display: true, position: 'top', align: 'end', labels: { boxWidth: 12, usePointStyle: true, font: { size: 11 } } },
            tooltip: {
              backgroundColor: 'rgba(16, 22, 26, 0.92)',
              titleFont: { family: 'ui-monospace, monospace', size: 11 },
              bodyFont: { family: 'ui-monospace, monospace', size: 11.5 },
              callbacks: {
                title: function (items) { return items.length ? fmtTime(items[0].parsed.x, 0) + ' UTC' : ''; }
              }
            }
          },
          scales: {
            x: {
              type: 'linear',
              min: view[0],
              max: view[1],
              grid: { color: '#edf2f7' },
              ticks: {
                maxTicksLimit: 9,
                font: { size: 11 },
                color: '#5c7080',
                callback: function (v) { return fmtTime(v, spanMs); }
              }
            },
            left: {
              type: 'linear',
              position: 'left',
              min: state.yBounds.telemLeft ? state.yBounds.telemLeft[0] : undefined,
              max: state.yBounds.telemLeft ? state.yBounds.telemLeft[1] : undefined,
              afterFit: function (s) { s.width = Math.max(s.width, 64); },
              grid: { color: '#edf2f7' },
              ticks: { includeBounds: !state.yBounds.telemLeft, font: { size: 11 }, color: '#184a90' },
              title: { display: true, text: 'Pressure (psia)', font: { size: 11, weight: '600' }, color: '#184a90' }
            },
            right: {
              type: 'linear',
              position: 'right',
              display: hasRight,
              afterFit: function (s) { if (hasRight) s.width = Math.max(s.width, 58); },
              grid: { drawOnChartArea: false },
              min: state.yBounds.telemRight ? state.yBounds.telemRight[0] : undefined,
              max: state.yBounds.telemRight ? state.yBounds.telemRight[1] : undefined,
              suggestedMin: state.yBounds.telemRight ? undefined : 0,
              suggestedMax: (!state.yBounds.telemRight && rightUnits.length === 1 && rightUnits[0] === '%') ? 100 : undefined,
              ticks: { includeBounds: !state.yBounds.telemRight, font: { size: 11 }, color: '#d9822b' },
              title: { display: hasRight, text: rightUnits.join(' / '), font: { size: 11, weight: '600' }, color: '#d9822b' }
            }
          }
        }
      });

      drawScrubber();
    }

    function drawScrubber() {
      if (!root || !payload || !payload.overview) return;
      var canvas = root.querySelector('#qv-scrubber-canvas');
      if (!canvas) return;
      var rect = canvas.parentElement.getBoundingClientRect();
      var w = Math.max(200, Math.round(rect.width)), h = Math.max(40, Math.round(rect.height));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, w, h);

      var b = payload.bounds_ms, v = payload.view_ms || b;
      var span = Math.max(1, b[1] - b[0]);
      var xOf = function (ms) { return ((ms - b[0]) / span) * w; };

      var ov = payload.overview;
      var st = arrayStats(ov.dh_pressure_psia);
      if (st.min !== null && st.max > st.min) {
        ctx.strokeStyle = '#8a9ba8';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        var started = false;
        for (var i = 0; i < ov.t_ms.length; i++) {
          var val = ov.dh_pressure_psia[i];
          if (val === null || !isFinite(val)) continue;
          var x = xOf(ov.t_ms[i]);
          var y = h - 6 - ((val - st.min) / (st.max - st.min)) * (h - 14);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      if (payload.shut_ins) {
        ctx.fillStyle = 'rgba(219, 55, 55, 0.35)';
        payload.shut_ins.forEach(function (si) {
          var x0 = xOf(si.start_ms), x1 = xOf(si.end_ms);
          ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
        });
      }

      var vx0 = Math.max(0, xOf(v[0])), vx1 = Math.min(w, xOf(v[1]));
      ctx.fillStyle = 'rgba(45, 114, 210, 0.16)';
      ctx.fillRect(vx0, 0, Math.max(4, vx1 - vx0), h);
      ctx.strokeStyle = '#2d72d2';
      ctx.lineWidth = 2;
      ctx.strokeRect(vx0, 1, Math.max(4, vx1 - vx0), h - 2);

      root.querySelector('#qv-scrub-range').textContent =
        fmtTime(v[0], 0) + ' UTC  \u2192  ' + fmtTime(v[1], 0) + ' UTC';
    }

    function populateEventDropdown() {
      if (!root || !payload) return;
      var sel = root.querySelector('#qv-event-jump');
      var opts = ['<option value="">Jump to Shut-in / Well Test...</option>'];
      (payload.shut_ins || []).forEach(function (si, idx) {
        opts.push('<option value="si:' + idx + '">Shut-in PBU · ' + fmtTime(si.start_ms, 0) + ' (' + si.duration_hr + 'h)</option>');
      });
      (payload.tests || []).forEach(function (t, idx) {
        opts.push('<option value="wt:' + idx + '">Well Test #' + (idx + 1) + ' · ' + fmtTime(t.t_mid_ms, 86400000 * 30) + ' (' + fmtNum(t.oil_rate_vfm_stbd, 0) + ' STB/d)</option>');
      });
      sel.innerHTML = opts.join('');
    }

    function updateRangeButtons() {
      if (!root) return;
      root.querySelectorAll('[data-qrange]').forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-qrange') === state.rangeKey);
      });
    }

    function applyRangePreset(key) {
      state.rangeKey = key;
      updateRangeButtons();
      if (!payload || !payload.bounds_ms) {
        state.viewStartMs = null; state.viewEndMs = null;
        fetchTelemetry();
        return;
      }
      var b = payload.bounds_ms;
      if (key === 'all') {
        state.viewStartMs = null; state.viewEndMs = null;
      } else {
        var days = key === '7d' ? 7 : (key === '30d' ? 30 : 90);
        state.viewEndMs = b[1];
        state.viewStartMs = Math.max(b[0], b[1] - days * 86400000);
      }
      fetchTelemetry();
    }

    async function fetchTelemetry() {
      var well = WB.state.currentWell;
      if (!root || !well) return;
      root.querySelector('#qv-well-tag').textContent = well;
      var mySeq = ++loadSeq;
      setLoading(true);
      var m = WB.state.activeModel || 'synth_stacked_v2';
      var url = '/api/monitoring?name=' + encodeURIComponent(well) + '&res=auto&max_pts=1600&model=' + encodeURIComponent(m);
      if (state.viewStartMs !== null && state.viewEndMs !== null) {
        url += '&start_ms=' + Math.round(state.viewStartMs) + '&end_ms=' + Math.round(state.viewEndMs);
      }
      try {
        var data = await WB.api(url);
        if (mySeq !== loadSeq) return;
        payload = data;
        setLoading(false);
        populateEventDropdown();
        renderKPIs();
        renderCharts();
      } catch (err) {
        if (mySeq !== loadSeq) return;
        setLoading(false);
        WB.toast('Monitoring load error: ' + err.message, 'error');
      }
    }

    function exportVisibleCsv() {
      if (!payload || !payload.t_ms) return;
      var keys = ['oil_rate_vfm_stbd', 'water_rate_vfm_stbd', 'gas_rate_vfm_mscfd', 'liquid_rate_vfm_stbd',
                  'dh_pressure_psia', 'whp_psia', 'whp_downstream_psia', 'choke_opening_pct', 'gas_lift_rate_mscfd', 'dh_temperature_degf']
        .filter(function (k) { return payload.series && payload.series[k]; });
      var rows = ['timestamp_utc,' + keys.join(',')];
      for (var i = 0; i < payload.t_ms.length; i++) {
        var vals = keys.map(function (k) {
          var v = payload.series[k][i];
          return (v === null || v === undefined) ? '' : v;
        });
        rows.push(new Date(payload.t_ms[i]).toISOString() + ',' + vals.join(','));
      }
      var blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (payload.well_name || 'well') + '_' + (payload.vfm_model || 'vfm') + '_monitoring.csv';
      a.click();
    }

    function renderPills() {
      if (!root) return;
      root.querySelector('#qv-prod-pills').innerHTML = PROD_SERIES.map(function (s) {
        return '<button type="button" class="qv-pill' + (state.prodOn[s.key] ? ' on' : '') + '" data-prod="' + s.key + '">' +
          '<span class="qv-dot" style="background:' + s.color + '"></span>' + WB.esc(s.label) + '</button>';
      }).join('');
      root.querySelector('#qv-telem-pills').innerHTML = TELEM_SERIES.map(function (s) {
        return '<button type="button" class="qv-pill' + (state.telemOn[s.key] ? ' on' : '') + '" data-telem="' + s.key + '">' +
          '<span class="qv-dot" style="background:' + s.color + '"></span>' + WB.esc(s.label) + '</button>';
      }).join('');

      root.querySelectorAll('[data-prod]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var k = btn.getAttribute('data-prod');
          state.prodOn[k] = !state.prodOn[k];
          renderPills();
          renderCharts();
        });
      });
      root.querySelectorAll('[data-telem]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var k = btn.getAttribute('data-telem');
          state.telemOn[k] = !state.telemOn[k];
          renderPills();
          renderCharts();
        });
      });
    }

    function buildSkeleton(el) {
      el.innerHTML = [
        '<div class="qv-root">',
          '<div class="qv-topbar">',
            '<div class="qv-bar-left">',
              '<span id="qv-well-tag" class="qv-well-tag">&mdash;</span>',
              '<div class="qv-seg">',
                '<button type="button" data-qrange="7d">7D</button>',
                '<button type="button" data-qrange="30d">30D</button>',
                '<button type="button" data-qrange="90d">90D</button>',
                '<button type="button" data-qrange="all" class="active">ALL</button>',
              '</div>',
              '<select id="qv-event-jump" class="qv-select"><option value="">Jump to Shut-in / Well Test...</option></select>',
            '</div>',
            '<div class="qv-bar-right">',
              '<div class="qv-seg" title="Plot drag tool: Box Zoom selects region; Pan shifts plot">',
                '<button type="button" data-pmode="box" class="active">⬚ Box Zoom</button>',
                '<button type="button" data-pmode="pan">✋ Pan</button>',
              '</div>',
              '<label class="qv-check"><input type="checkbox" id="qv-chk-si" checked> Shut-ins</label>',
              '<label class="qv-check"><input type="checkbox" id="qv-chk-wt" checked> Well Tests</label>',
              '<button type="button" id="qv-reset-btn" class="qv-btn">Reset Axes</button>',
              '<button type="button" id="qv-csv-btn" class="qv-btn">Export CSV</button>',
              '<span id="qv-status" class="qv-meta-pill">Ready</span>',
            '</div>',
          '</div>',

          '<div class="qv-kpi-grid">',
            '<div class="qv-kpi">',
              '<span class="qv-kpi-lbl">Window Avg Oil Rate</span>',
              '<span id="qv-kpi-oil" class="qv-kpi-val">&mdash;</span>',
              '<span id="qv-kpi-oil-sub" class="qv-kpi-sub">&mdash;</span>',
            '</div>',
            '<div class="qv-kpi">',
              '<span class="qv-kpi-lbl">Downhole Pressure (PDG)</span>',
              '<span id="qv-kpi-dhp" class="qv-kpi-val">&mdash;</span>',
              '<span id="qv-kpi-dhp-sub" class="qv-kpi-sub">&mdash;</span>',
            '</div>',
            '<div class="qv-kpi">',
              '<span class="qv-kpi-lbl">Wellhead &amp; Choke</span>',
              '<span id="qv-kpi-whp" class="qv-kpi-val">&mdash;</span>',
              '<span id="qv-kpi-whp-sub" class="qv-kpi-sub">&mdash;</span>',
            '</div>',
            '<div class="qv-kpi">',
              '<span class="qv-kpi-lbl">Flowing Uptime</span>',
              '<span id="qv-kpi-up" class="qv-kpi-val">&mdash;</span>',
              '<span id="qv-kpi-up-sub" class="qv-kpi-sub">&mdash;</span>',
            '</div>',
          '</div>',

          '<div class="qv-card">',
            '<div class="qv-card-hdr">',
              '<span class="qv-card-title">Production &amp; Active VFM Interpolation <span id="qv-vfm-badge" class="qv-model-badge">Model: synth_stacked_v2</span></span>',
              '<div id="qv-prod-pills" class="qv-pills"></div>',
            '</div>',
            '<div class="qv-canvas-wrap" style="height:295px"><canvas id="qv-prod-canvas"></canvas></div>',
          '</div>',

          '<div class="qv-card">',
            '<div class="qv-card-hdr">',
              '<span class="qv-card-title">Downhole &amp; Wellhead Telemetry</span>',
              '<div id="qv-telem-pills" class="qv-pills"></div>',
            '</div>',
            '<div class="qv-canvas-wrap" style="height:265px"><canvas id="qv-telem-canvas"></canvas></div>',
          '</div>',

          '<div class="qv-scrubber-card">',
            '<div class="qv-scrubber-wrap"><canvas id="qv-scrubber-canvas"></canvas></div>',
            '<div class="qv-scrubber-foot">',
              '<span id="qv-scrub-range">&mdash;</span>',
              '<span><strong>Drag Left/Right Y-axis</strong> = Shift independently · <strong>Scroll on any axis</strong> (or drag axis ends) = Rescale · <strong>Drag plot</strong> = Box Zoom / Pan · <strong>Double-click</strong> = Reset</span>',
            '</div>',
          '</div>',
        '</div>'
      ].join('');

      renderPills();

      el.querySelectorAll('[data-qrange]').forEach(function (btn) {
        btn.addEventListener('click', function () { applyRangePreset(btn.getAttribute('data-qrange')); });
      });
      el.querySelectorAll('[data-pmode]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          state.plotMode = btn.getAttribute('data-pmode');
          el.querySelectorAll('[data-pmode]').forEach(function (b) { b.classList.toggle('active', b === btn); });
        });
      });
      el.querySelector('#qv-reset-btn').addEventListener('click', function () {
        state.yBounds = { prod: null, telemLeft: null, telemRight: null };
        applyRangePreset('all');
      });
      el.querySelector('#qv-csv-btn').addEventListener('click', exportVisibleCsv);
      el.querySelector('#qv-chk-si').addEventListener('change', function (e) { state.showShutins = e.target.checked; renderCharts(); });
      el.querySelector('#qv-chk-wt').addEventListener('change', function (e) { state.showTests = e.target.checked; renderCharts(); });

      el.querySelector('#qv-event-jump').addEventListener('change', function (e) {
        var val = e.target.value;
        if (!val || !payload) return;
        var parts = val.split(':');
        var idx = Number(parts[1]);
        if (parts[0] === 'si' && payload.shut_ins[idx]) {
          var si = payload.shut_ins[idx];
          syncXView([si.start_ms - 6 * 3600000, si.end_ms + 6 * 3600000], true);
        } else if (parts[0] === 'wt' && payload.tests[idx]) {
          var wt = payload.tests[idx];
          syncXView([wt.t_mid_ms - 3 * 86400000, wt.t_mid_ms + 3 * 86400000], true);
        }
      });

      prodZoomHandle = wireCanvas(el.querySelector('#qv-prod-canvas'), function () { return prodChart; }, 'prod');
      telemZoomHandle = wireCanvas(el.querySelector('#qv-telem-canvas'), function () { return telemChart; }, 'telem');

      var scCanvas = el.querySelector('#qv-scrubber-canvas');
      scCanvas.addEventListener('mousedown', function (e) {
        if (!payload || !payload.bounds_ms) return;
        var rect = scCanvas.getBoundingClientRect();
        brushDrag = { x0: e.clientX - rect.left, x1: e.clientX - rect.left, w: rect.width };
      });
      window.addEventListener('mousemove', function (e) {
        if (!brushDrag) return;
        var rect = scCanvas.getBoundingClientRect();
        brushDrag.x1 = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      });
      window.addEventListener('mouseup', function () {
        if (!brushDrag || !payload || !payload.bounds_ms) return;
        var bd = brushDrag;
        brushDrag = null;
        if (Math.abs(bd.x1 - bd.x0) < 8) return;
        var b = payload.bounds_ms, span = b[1] - b[0];
        syncXView([b[0] + (Math.min(bd.x0, bd.x1) / bd.w) * span, b[0] + (Math.max(bd.x0, bd.x1) / bd.w) * span], true);
      });
    }

    // Automatically refresh Monitoring whenever the user activates or finishes computing a different VFM model!
    WB.on('activeModel', function () {
      if (root && root.classList.contains('active')) fetchTelemetry();
    });

    WB.registerTab({
      id: 'monitoring',
      title: 'Field Monitoring',
      order: 2,
      render: function (container) {
        root = container;
        buildSkeleton(container);
      },
      onShow: function () {
        fetchTelemetry();
      },
      onWellChange: function () {
        state.rangeKey = 'all';
        state.viewStartMs = null;
        state.viewEndMs = null;
        state.yBounds = { prod: null, telemLeft: null, telemRight: null };
        updateRangeButtons();
        if (root && root.classList.contains('active')) fetchTelemetry();
      }
    });
  }

  boot();
})();
