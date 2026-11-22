(function () {
  'use strict';

  function boot() {
    if (!window.WB || typeof window.WB.registerTab !== 'function') {
      setTimeout(boot, 25);
      return;
    }

    var WB = window.WB;

    var STYLE_ID = 'tm-injected-styles';
    if (!document.getElementById(STYLE_ID)) {
      var styleEl = document.createElement('style');
      styleEl.id = STYLE_ID;
      styleEl.textContent = [
        '.tm-root { display: flex; flex-direction: column; gap: 14px; color: var(--fg, #1f2937); font-size: 13px; }',
        '.tm-card { background: var(--bg, #ffffff); border: 1px solid var(--border, #d1d5db); border-radius: 8px; padding: 14px 16px; }',
        '.tm-card-header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }',
        '.tm-title { margin: 0; font-size: 15px; font-weight: 600; }',
        '.tm-muted { color: var(--muted, #6b7280); font-size: 12px; }',
        '.tm-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }',
        '.tm-grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }',
        '.tm-btn { cursor: pointer; border: 1px solid var(--border, #d1d5db); background: var(--bg, #ffffff); color: inherit; border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 500; }',
        '.tm-btn:hover { background: rgba(128,128,128,0.08); }',
        '.tm-btn:disabled { opacity: 0.55; cursor: not-allowed; }',
        '.tm-btn-primary { background: #2563eb; color: #ffffff; border-color: #2563eb; }',
        '.tm-btn-primary:hover { background: #1d4ed8; }',
        '.tm-seg { display: inline-flex; border: 1px solid var(--border, #d1d5db); border-radius: 6px; overflow: hidden; }',
        '.tm-seg button { border: none; background: var(--bg, #ffffff); color: inherit; padding: 5px 11px; font-size: 12px; cursor: pointer; }',
        '.tm-seg button + button { border-left: 1px solid var(--border, #d1d5db); }',
        '.tm-seg button.tm-active { background: #2563eb; color: #ffffff; font-weight: 600; }',
        '.tm-select, .tm-input { border: 1px solid var(--border, #d1d5db); background: var(--bg, #ffffff); color: inherit; border-radius: 6px; padding: 5px 8px; font-size: 12px; }',
        '.tm-badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 500; border: 1px solid var(--border, #d1d5db); }',
        '.tm-badge-ready { background: #dcfce7; color: #166534; border-color: #bbf7d0; }',
        '.tm-badge-running { background: #fef9c3; color: #854d0e; border-color: #fde68a; }',
        '.tm-badge-error { background: #fee2e2; color: #991b1b; border-color: #fecaca; }',
        '.tm-badge-idle { background: rgba(128,128,128,0.1); color: var(--muted, #6b7280); }',
        '.tm-table-wrap { overflow-x: auto; max-width: 100%; }',
        '.tm-table { width: 100%; border-collapse: collapse; font-size: 12px; }',
        '.tm-table th, .tm-table td { border: 1px solid var(--border, #e5e7eb); padding: 5px 7px; text-align: right; white-space: nowrap; }',
        '.tm-table th:first-child, .tm-table td:first-child, .tm-table th.tm-left, .tm-table td.tm-left { text-align: left; }',
        '.tm-table th { background: rgba(128,128,128,0.06); font-weight: 600; }',
        '.tm-table th.tm-sortable { cursor: pointer; user-select: none; }',
        '.tm-table th.tm-sortable:hover { background: rgba(128,128,128,0.12); }',
        '.tm-table tr.tm-clickable { cursor: pointer; }',
        '.tm-table tr.tm-clickable:hover { background: rgba(37,99,235,0.06); }',
        '.tm-table tr.tm-row-selected { background: rgba(37,99,235,0.12); }',
        '.tm-best { font-weight: 700; }',
        '.tm-chart-box { position: relative; height: 290px; width: 100%; }',
        '.tm-chart-box-tall { position: relative; height: 340px; width: 100%; }',
        '.tm-heat-cell { text-align: center !important; font-variant-numeric: tabular-nums; min-width: 54px; }',
        '.tm-checks { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 8px 0 10px; }',
        '.tm-check-item { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer; }',
        '.tm-corr-wrap { overflow: auto; max-height: 540px; max-width: 100%; }',
        '.tm-corr-table { border-collapse: collapse; font-size: 11px; }',
        '.tm-corr-table th, .tm-corr-table td { border: 1px solid var(--border, #e5e7eb); }',
        '.tm-corr-col-th { height: 125px; width: 32px; min-width: 32px; padding: 4px 2px; vertical-align: bottom; text-align: left; background: var(--bg, #ffffff); position: sticky; top: 0; z-index: 2; }',
        '.tm-corr-col-lbl { display: inline-block; transform: rotate(-60deg); transform-origin: bottom left; white-space: nowrap; width: 16px; margin-left: 12px; margin-bottom: 4px; font-weight: 600; }',
        '.tm-corr-row-th { padding: 4px 8px; text-align: right; white-space: nowrap; font-weight: 600; background: var(--bg, #ffffff); position: sticky; left: 0; z-index: 1; }',
        '.tm-corr-corner { position: sticky; top: 0; left: 0; z-index: 3; background: var(--bg, #ffffff); }',
        '.tm-corr-cell { width: 32px; min-width: 32px; height: 26px; text-align: center; cursor: pointer; font-variant-numeric: tabular-nums; padding: 0 2px; transition: opacity 0.12s ease; }',
        '.tm-corr-cell:hover { outline: 2px solid #111827; outline-offset: -2px; }',
        '.tm-corr-cell.tm-corr-sel { outline: 2px solid #2563eb; outline-offset: -2px; }',
        '.tm-corr-cell.tm-dim { opacity: 0.18; }',
        '.tm-pair-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; max-height: 340px; overflow-y: auto; }',
        '.tm-pair-item { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 5px 8px; border: 1px solid var(--border, #e5e7eb); border-radius: 5px; cursor: pointer; font-size: 12px; }',
        '.tm-pair-item:hover { background: rgba(37,99,235,0.06); }',
        '.tm-pair-item.tm-pair-sel { border-color: #2563eb; background: rgba(37,99,235,0.1); font-weight: 600; }',
        '.tm-empty { padding: 18px; text-align: center; color: var(--muted, #6b7280); border: 1px dashed var(--border, #d1d5db); border-radius: 6px; }'
      ].join('\n');
      document.head.appendChild(styleEl);
    }

    function statusBadgeHtml(status) {
      var s = String(status || 'unknown');
      var low = s.toLowerCase();
      var cls = 'tm-badge-idle';
      if (low.indexOf('ready') === 0) cls = 'tm-badge-ready';
      else if (low.indexOf('running') === 0) cls = 'tm-badge-running';
      else if (low.indexOf('error') === 0) cls = 'tm-badge-error';
      return '<span class="tm-badge ' + cls + '">' + WB.esc(s) + '</span>';
    }

    function isTerminalStatus(status) {
      var low = String(status || '').toLowerCase();
      return low.indexOf('ready') === 0 || low.indexOf('error') === 0;
    }

    function isRunningStatus(status) {
      return String(status || '').toLowerCase().indexOf('running') === 0;
    }

    function isFiniteNum(v) {
      return typeof v === 'number' && isFinite(v);
    }

    function lerpChannel(a, b, t) {
      return Math.round(a + (b - a) * t);
    }

    function greenToRedColor(val, min, max) {
      if (!isFiniteNum(val)) return { bg: '#e5e7eb', fg: '#4b5563' };
      var t = max > min ? (val - min) / (max - min) : 0;
      t = Math.max(0, Math.min(1, t));
      // Green (#bbf7d0 -> 187,247,208) -> Yellow (#fef08a -> 254,240,138) -> Red (#fca5a5 -> 252,165,165)
      var r, g, b;
      if (t <= 0.5) {
        var u = t * 2;
        r = lerpChannel(134, 253, u);
        g = lerpChannel(239, 224, u);
        b = lerpChannel(172, 71, u);
      } else {
        var u2 = (t - 0.5) * 2;
        r = lerpChannel(253, 248, u2);
        g = lerpChannel(224, 113, u2);
        b = lerpChannel(71, 113, u2);
      }
      return { bg: 'rgb(' + r + ',' + g + ',' + b + ')', fg: '#111827' };
    }

    function divergingColor(val) {
      if (!isFiniteNum(val)) return { bg: '#e5e7eb', fg: '#4b5563' };
      var clamped = Math.max(-1, Math.min(1, val));
      var t = Math.abs(clamped);
      var r, g, b;
      if (clamped < 0) {
        // 0 white (#ffffff) -> -1 blue (#2166ac: 33,102,172)
        r = lerpChannel(255, 33, t);
        g = lerpChannel(255, 102, t);
        b = lerpChannel(255, 172, t);
      } else {
        // 0 white (#ffffff) -> +1 red (#b2182b: 178,24,43)
        r = lerpChannel(255, 178, t);
        g = lerpChannel(255, 24, t);
        b = lerpChannel(255, 43, t);
      }
      var fg = t > 0.55 ? '#ffffff' : '#111827';
      return { bg: 'rgb(' + r + ',' + g + ',' + b + ')', fg: fg };
    }

    function getOverviewWells() {
      var ov = WB.state && WB.state.overview;
      var raw = ov && Array.isArray(ov.wells) ? ov.wells : [];
      var out = [];
      for (var i = 0; i < raw.length; i++) {
        var w = raw[i];
        var name = typeof w === 'string' ? w : (w && (w.name || w.well_name));
        if (name) out.push(name);
      }
      return out;
    }

    // =========================================================================
    // TAB 4: MODEL COMPARISON
    // =========================================================================
    var mState = {
      container: null,
      rendered: false,
      modelsData: null,
      benchmarkData: null,
      wellData: null,
      protocol: 'walk_forward',
      metric: 'liquid_rate_stbd_mape',
      sortKey: 'liquid_rate_stbd_mape',
      sortDir: 'asc',
      selectedModel: null,
      checkedModels: null,
      dailyPhase: 'liquid',
      unanchored: false,
      currentWell: null,
      benchTimer: null,
      modelsTimer: null,
      loadingDaily: false
    };

    var METRIC_OPTIONS = [
      { key: 'liquid_rate_stbd_mae', label: 'Liquid MAE (STB/d)', phase: 'liquid', stat: 'mae' },
      { key: 'liquid_rate_stbd_mape', label: 'Liquid MAPE (%)', phase: 'liquid', stat: 'mape' },
      { key: 'oil_rate_stbd_mae', label: 'Oil MAE (STB/d)', phase: 'oil', stat: 'mae' },
      { key: 'oil_rate_stbd_mape', label: 'Oil MAPE (%)', phase: 'oil', stat: 'mape' },
      { key: 'water_rate_stbd_mae', label: 'Water MAE (STB/d)', phase: 'water', stat: 'mae' },
      { key: 'water_rate_stbd_mape', label: 'Water MAPE (%)', phase: 'water', stat: 'mape' },
      { key: 'gas_rate_mscfd_mae', label: 'Gas MAE (Mscf/d)', phase: 'gas', stat: 'mae' },
      { key: 'gas_rate_mscfd_mape', label: 'Gas MAPE (%)', phase: 'gas', stat: 'mape' }
    ];

    var SUMMARY_PHASES = [
      { key: 'liquid', label: 'Liquid', prefix: 'liquid_rate_stbd_' },
      { key: 'oil', label: 'Oil', prefix: 'oil_rate_stbd_' },
      { key: 'water', label: 'Water', prefix: 'water_rate_stbd_' },
      { key: 'gas', label: 'Gas', prefix: 'gas_rate_mscfd_' }
    ];

    function getModelsMap() {
      var map = {};
      var list = (mState.modelsData && mState.modelsData.models) || [];
      for (var i = 0; i < list.length; i++) {
        map[list[i].name] = list[i];
      }
      return map;
    }

    function ensureDefaultCheckedModels() {
      var list = (mState.modelsData && mState.modelsData.models) || [];
      var active = (mState.modelsData && mState.modelsData.active) || WB.state.activeModel;
      if (!mState.checkedModels) {
        mState.checkedModels = {};
        for (var i = 0; i < list.length; i++) {
          var name = list[i].name;
          if (name === 'physics_wls' || name === active) {
            mState.checkedModels[name] = true;
          }
        }
        if (Object.keys(mState.checkedModels).length === 0 && list.length > 0) {
          mState.checkedModels[list[0].name] = true;
        }
      }
    }

    function getCheckedModelNames() {
      var list = (mState.modelsData && mState.modelsData.models) || [];
      var checked = mState.checkedModels || {};
      var out = [];
      for (var i = 0; i < list.length; i++) {
        if (checked[list[i].name]) out.push(list[i].name);
      }
      return out;
    }

    function stopBenchPoll() {
      if (mState.benchTimer) {
        clearInterval(mState.benchTimer);
        mState.benchTimer = null;
      }
    }

    function stopModelsPoll() {
      if (mState.modelsTimer) {
        clearInterval(mState.modelsTimer);
        mState.modelsTimer = null;
      }
    }

    function startBenchPoll() {
      if (mState.benchTimer) return;
      mState.benchTimer = setInterval(function () {
        WB.api('/api/models').then(function (md) {
          mState.modelsData = md;
          WB.emit('models', md);
          if (!isRunningStatus(md.benchmark_status)) {
            stopBenchPoll();
            WB.api('/api/benchmark').then(function (bd) {
              mState.benchmarkData = bd;
              renderModelsTab();
            }).catch(function (err) {
              WB.toast(err.message, 'error');
              renderModelsTab();
            });
          } else {
            renderModelsTab();
          }
        }).catch(function () {});
      }, 5000);
    }

    function startModelsPoll(targetNames, onAllDone) {
      stopModelsPoll();
      mState.modelsTimer = setInterval(function () {
        WB.api('/api/models').then(function (md) {
          mState.modelsData = md;
          WB.emit('models', md);
          var map = getModelsMap();
          var anyRunning = false;
          var allDone = true;
          for (var i = 0; i < md.models.length; i++) {
            if (isRunningStatus(md.models[i].status)) anyRunning = true;
          }
          if (targetNames && targetNames.length) {
            for (var j = 0; j < targetNames.length; j++) {
              var info = map[targetNames[j]];
              if (!info || !isTerminalStatus(info.status)) {
                allDone = false;
              }
            }
          } else {
            allDone = !anyRunning;
          }
          if (allDone) {
            stopModelsPoll();
            if (onAllDone) onAllDone();
            else renderModelsTab();
          } else {
            renderModelsTab();
          }
        }).catch(function () {});
      }, 3000);
    }

    function checkRunningModelsAndPoll() {
      var list = (mState.modelsData && mState.modelsData.models) || [];
      for (var i = 0; i < list.length; i++) {
        if (isRunningStatus(list[i].status)) {
          startModelsPoll(null, null);
          break;
        }
      }
      if (mState.modelsData && isRunningStatus(mState.modelsData.benchmark_status)) {
        startBenchPoll();
      }
    }

    function loadDailyComparison(onlyIfReady) {
      var well = mState.currentWell || WB.state.currentWell;
      var names = getCheckedModelNames();
      if (!well || names.length === 0) {
        mState.wellData = null;
        renderModelsTab();
        return;
      }
      var map = getModelsMap();
      var queryNames = [];
      for (var i = 0; i < names.length; i++) {
        var info = map[names[i]];
        if (!onlyIfReady || (info && String(info.status || '').toLowerCase().indexOf('ready') === 0)) {
          queryNames.push(names[i]);
        }
      }
      if (queryNames.length === 0) {
        mState.wellData = null;
        renderModelsTab();
        return;
      }
      mState.loadingDaily = true;
      WB.api('/api/models/well?name=' + encodeURIComponent(well) + '&models=' + encodeURIComponent(queryNames.join(',')))
        .then(function (data) {
          mState.loadingDaily = false;
          mState.wellData = data;
          renderModelsTab();
        })
        .catch(function (err) {
          mState.loadingDaily = false;
          WB.toast(err.message, 'error');
          renderModelsTab();
        });
    }

    function loadModelsTabData(refreshDaily) {
      Promise.all([
        WB.api('/api/models'),
        WB.api('/api/benchmark')
      ]).then(function (res) {
        mState.modelsData = res[0];
        mState.benchmarkData = res[1];
        if (!mState.currentWell) {
          var wells = getOverviewWells();
          mState.currentWell = WB.state.currentWell || (wells.length ? wells[0] : null);
        }
        ensureDefaultCheckedModels();
        checkRunningModelsAndPoll();
        if (refreshDaily !== false) {
          loadDailyComparison(true);
        } else {
          renderModelsTab();
        }
      }).catch(function (err) {
        WB.toast('Failed to load model comparison: ' + err.message, 'error');
        renderModelsTab();
      });
    }

    function getFilteredSummary() {
      var bd = mState.benchmarkData;
      var raw = (bd && Array.isArray(bd.summary)) ? bd.summary : [];
      var out = [];
      for (var i = 0; i < raw.length; i++) {
        var row = raw[i];
        if (!row.cv_mode || row.cv_mode === mState.protocol) {
          out.push(row);
        }
      }
      return out;
    }

    function getFilteredPerWell() {
      var bd = mState.benchmarkData;
      var raw = (bd && Array.isArray(bd.per_well)) ? bd.per_well : [];
      var out = [];
      for (var i = 0; i < raw.length; i++) {
        var row = raw[i];
        if (!row.cv_mode || row.cv_mode === mState.protocol) {
          out.push(row);
        }
      }
      return out;
    }

    function getFilteredPoints(modelName) {
      var bd = mState.benchmarkData;
      var raw = (bd && Array.isArray(bd.points)) ? bd.points : [];
      var out = [];
      for (var i = 0; i < raw.length; i++) {
        var p = raw[i];
        var pm = p.model || p.name;
        if ((!p.cv_mode || p.cv_mode === mState.protocol) && (!modelName || pm === modelName)) {
          out.push(p);
        }
      }
      return out;
    }

    function extractPointXY(p) {
      var yTrue = p.y_true !== undefined ? p.y_true
        : (p.true_liquid !== undefined ? p.true_liquid
        : (p.liquid_true !== undefined ? p.liquid_true
        : (p.actual_liquid !== undefined ? p.actual_liquid
        : (p.liquid_actual !== undefined ? p.liquid_actual
        : (p.liquid_obs !== undefined ? p.liquid_obs
        : (p.obs_liquid !== undefined ? p.obs_liquid
        : (p.liquid_rate_stbd !== undefined ? p.liquid_rate_stbd
        : (p.test_liquid !== undefined ? p.test_liquid
        : (p.liquid !== undefined ? p.liquid
        : (p.actual !== undefined ? p.actual : p.true))))))))));
      var yPred = p.y_pred !== undefined ? p.y_pred
        : (p.pred_liquid !== undefined ? p.pred_liquid
        : (p.liquid_pred !== undefined ? p.liquid_pred
        : (p.predicted_liquid !== undefined ? p.predicted_liquid
        : (p.liquid_predicted !== undefined ? p.liquid_predicted
        : (p.pred_liquid_rate_stbd !== undefined ? p.pred_liquid_rate_stbd
        : (p.liquid_vfm !== undefined ? p.liquid_vfm
        : (p.vfm_liquid !== undefined ? p.vfm_liquid
        : (p.pred !== undefined ? p.pred : p.predicted))))))));
      return {
        well: p.well_name || p.well || 'Unknown',
        x: yTrue,
        y: yPred
      };
    }

    function renderModelsTab() {
      var root = mState.container;
      if (!root) return;

      var md = mState.modelsData || {};
      var bd = mState.benchmarkData || {};
      var modelsMap = getModelsMap();
      var modelsList = md.models || [];
      var benchStatus = md.benchmark_status || bd.status || 'idle';
      var benchRunning = isRunningStatus(benchStatus);

      var summaryRows = getFilteredSummary();
      var activeModel = md.active || WB.state.activeModel;

      // Ensure selectedModel is valid for current summary if possible
      var hasSel = false;
      for (var sIdx = 0; sIdx < summaryRows.length; sIdx++) {
        if ((summaryRows[sIdx].model || summaryRows[sIdx].name) === mState.selectedModel) {
          hasSel = true;
          break;
        }
      }
      if (!hasSel && summaryRows.length > 0) {
        var fallback = null;
        for (var fIdx = 0; fIdx < summaryRows.length; fIdx++) {
          var mName = summaryRows[fIdx].model || summaryRows[fIdx].name;
          if (mName === activeModel) { fallback = mName; break; }
          if (!fallback && mName === 'physics_wls') fallback = mName;
        }
        mState.selectedModel = fallback || (summaryRows[0].model || summaryRows[0].name);
      }

      // Build HTML
      var html = '<div class="tm-root">';

      // 1. Header card
      html += '<div class="tm-card">';
      html += '<div class="tm-card-header">';
      html += '<div class="tm-row">';
      html += '<h3 class="tm-title">Benchmark</h3>';
      html += statusBadgeHtml(benchStatus);
      html += '</div>';
      html += '<button id="tm-rerun-bench" class="tm-btn tm-btn-primary"' + (benchRunning ? ' disabled' : '') + '>' +
        (benchRunning ? 'Benchmark running...' : 'Re-run benchmark') + '</button>';
      html += '</div>';
      html += '<div class="tm-muted" style="margin-bottom:10px;">Walk-forward is strictly causal; LOTO leaves one test out; test_linear is hindcast-only (LOTO only).</div>';

      // 2. Protocol toggle & metric select
      html += '<div class="tm-row" style="justify-content:space-between;">';
      html += '<div class="tm-row">';
      html += '<span class="tm-muted">Protocol:</span>';
      html += '<div class="tm-seg" id="tm-proto-seg">';
      html += '<button data-proto="walk_forward" class="' + (mState.protocol === 'walk_forward' ? 'tm-active' : '') + '">Walk-forward</button>';
      html += '<button data-proto="loto" class="' + (mState.protocol === 'loto' ? 'tm-active' : '') + '">LOTO</button>';
      html += '</div>';
      html += '</div>';
      html += '<div class="tm-row">';
      html += '<label class="tm-muted" for="tm-metric-sel">Metric:</label>';
      html += '<select id="tm-metric-sel" class="tm-select">';
      for (var mo = 0; mo < METRIC_OPTIONS.length; mo++) {
        var opt = METRIC_OPTIONS[mo];
        html += '<option value="' + WB.esc(opt.key) + '"' + (mState.metric === opt.key ? ' selected' : '') + '>' + WB.esc(opt.label) + '</option>';
      }
      html += '</select>';
      html += '</div>';
      html += '</div>';
      html += '</div>'; // end header card

      // Benchmark content
      if (!bd.available || summaryRows.length === 0) {
        html += '<div class="tm-card"><div class="tm-empty">' +
          (bd.available === false ? 'No benchmark results available yet. Click "Re-run benchmark" to run.' : 'No summary rows for protocol "' + WB.esc(mState.protocol) + '".') +
          '</div></div>';
      } else {
        // Prepare summary columns and best values
        var metricCols = [];
        for (var sp = 0; sp < SUMMARY_PHASES.length; sp++) {
          var pInfo = SUMMARY_PHASES[sp];
          metricCols.push({ key: pInfo.prefix + 'mae', label: pInfo.label + ' MAE', kind: 'absmin' });
          metricCols.push({ key: pInfo.prefix + 'mape', label: pInfo.label + ' MAPE%', kind: 'absmin' });
          metricCols.push({ key: pInfo.prefix + 'bias', label: pInfo.label + ' Bias', kind: 'zero' });
        }

        var bestPerCol = {};
        for (var mc = 0; mc < metricCols.length; mc++) {
          var colDef = metricCols[mc];
          var bestVal = Infinity;
          for (var r = 0; r < summaryRows.length; r++) {
            var v = summaryRows[r][colDef.key];
            if (!isFiniteNum(v)) continue;
            var score = colDef.kind === 'zero' ? Math.abs(v) : v;
            if (score < bestVal) bestVal = score;
          }
          bestPerCol[colDef.key] = bestVal;
        }

        // Sort summaryRows
        var sortedSummary = summaryRows.slice().sort(function (a, b) {
          var sk = mState.sortKey;
          var dir = mState.sortDir === 'desc' ? -1 : 1;
          var va, vb;
          if (sk === 'model') {
            va = String(a.model || a.name || '');
            vb = String(b.model || b.name || '');
            return va.localeCompare(vb) * dir;
          }
          if (sk === 'family') {
            var ma = modelsMap[a.model || a.name];
            var mb = modelsMap[b.model || b.name];
            va = String((ma && ma.family) || a.family || '');
            vb = String((mb && mb.family) || b.family || '');
            return va.localeCompare(vb) * dir;
          }
          va = a[sk];
          vb = b[sk];
          var fa = isFiniteNum(va);
          var fb = isFiniteNum(vb);
          if (!fa && !fb) return 0;
          if (!fa) return 1;
          if (!fb) return -1;
          return (va - vb) * dir;
        });

        // 3. Summary table
        html += '<div class="tm-card">';
        html += '<div class="tm-card-header"><h3 class="tm-title">Cross-validation summary (' + WB.esc(mState.protocol) + ')</h3><span class="tm-muted">Click headers to sort; click a row to inspect scatter</span></div>';
        html += '<div class="tm-table-wrap"><table class="tm-table" id="tm-summary-table">';
        html += '<thead><tr>';
        var fixedHeaders = [
          { key: 'model', label: 'Model', cls: 'tm-left' },
          { key: 'family', label: 'Family', cls: 'tm-left' },
          { key: 'n_tests', label: 'n_tests', cls: '' }
        ];
        for (var fh = 0; fh < fixedHeaders.length; fh++) {
          var fHdr = fixedHeaders[fh];
          var arrow = mState.sortKey === fHdr.key ? (mState.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
          html += '<th class="tm-sortable ' + fHdr.cls + '" data-sort="' + WB.esc(fHdr.key) + '">' + WB.esc(fHdr.label) + arrow + '</th>';
        }
        for (var ch = 0; ch < metricCols.length; ch++) {
          var cHdr = metricCols[ch];
          var cArrow = mState.sortKey === cHdr.key ? (mState.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
          html += '<th class="tm-sortable" data-sort="' + WB.esc(cHdr.key) + '">' + WB.esc(cHdr.label) + cArrow + '</th>';
        }
        html += '</tr></thead><tbody>';

        for (var sr = 0; sr < sortedSummary.length; sr++) {
          var sRow = sortedSummary[sr];
          var mName = sRow.model || sRow.name || '';
          var mMeta = modelsMap[mName] || {};
          var fam = mMeta.family || sRow.family || '—';
          var isSel = mName === mState.selectedModel;
          html += '<tr class="tm-clickable' + (isSel ? ' tm-row-selected' : '') + '" data-model="' + WB.esc(mName) + '">';
          html += '<td class="tm-left">' + WB.esc(mName) + (mName === activeModel ? ' <span class="tm-badge tm-badge-ready">active</span>' : '') + '</td>';
          html += '<td class="tm-left">' + WB.esc(fam) + '</td>';
          html += '<td>' + (isFiniteNum(sRow.n_tests) ? WB.esc(String(sRow.n_tests)) : '—') + '</td>';
          for (var cm = 0; cm < metricCols.length; cm++) {
            var mcDef = metricCols[cm];
            var val = sRow[mcDef.key];
            var isBest = false;
            if (isFiniteNum(val) && isFiniteNum(bestPerCol[mcDef.key])) {
              var sc = mcDef.kind === 'zero' ? Math.abs(val) : val;
              if (Math.abs(sc - bestPerCol[mcDef.key]) < 1e-9) isBest = true;
            }
            var digits = mcDef.key.indexOf('_mape') >= 0 ? 1 : 1;
            html += '<td class="' + (isBest ? 'tm-best' : '') + '">' + WB.esc(WB.fmt(val, digits)) + '</td>';
          }
          html += '</tr>';
        }
        html += '</tbody></table></div></div>';

        // 4 & 6. Bar chart + Scatter chart side by side
        html += '<div class="tm-grid-2">';
        html += '<div class="tm-card">';
        html += '<div class="tm-card-header"><h3 class="tm-title">Selected metric ranking</h3><span class="tm-muted">' + WB.esc(mState.metric) + '</span></div>';
        html += '<div class="tm-chart-box"><canvas id="tm-bar-canvas"></canvas></div>';
        html += '</div>';

        html += '<div class="tm-card">';
        html += '<div class="tm-card-header"><h3 class="tm-title">True vs predicted liquid — ' + WB.esc(mState.selectedModel || 'none') + '</h3><span class="tm-muted">STB/d (dashed y = x)</span></div>';
        html += '<div class="tm-chart-box"><canvas id="tm-bench-scatter-canvas"></canvas></div>';
        html += '</div>';
        html += '</div>';

        // 5. Per-well heat table
        var perWellRows = getFilteredPerWell();
        html += '<div class="tm-card">';
        html += '<div class="tm-card-header"><h3 class="tm-title">Per-well liquid MAPE (%)</h3><span class="tm-muted">Green (low) to red (high)</span></div>';
        if (perWellRows.length === 0) {
          html += '<div class="tm-empty">No per-well metrics for this protocol.</div>';
        } else {
          var pwModels = [];
          var pwModelSeen = {};
          var pwWells = [];
          var pwWellSeen = {};
          var pwLookup = {};
          var pwMin = Infinity;
          var pwMax = -Infinity;

          for (var pw = 0; pw < perWellRows.length; pw++) {
            var pwr = perWellRows[pw];
            var pm = pwr.model || pwr.name;
            var pwn = pwr.well_name || pwr.well;
            if (!pm || !pwn) continue;
            if (!pwModelSeen[pm]) { pwModelSeen[pm] = true; pwModels.push(pm); }
            if (!pwWellSeen[pwn]) { pwWellSeen[pwn] = true; pwWells.push(pwn); }
            var lv = pwr.liquid_mape;
            if (!pwLookup[pm]) pwLookup[pm] = {};
            pwLookup[pm][pwn] = lv;
            if (isFiniteNum(lv)) {
              if (lv < pwMin) pwMin = lv;
              if (lv > pwMax) pwMax = lv;
            }
          }
          pwWells.sort();

          html += '<div class="tm-table-wrap"><table class="tm-table"><thead><tr><th class="tm-left">Model</th>';
          for (var wc = 0; wc < pwWells.length; wc++) {
            html += '<th>' + WB.esc(pwWells[wc]) + '</th>';
          }
          html += '</tr></thead><tbody>';
          for (var mr = 0; mr < pwModels.length; mr++) {
            var rmName = pwModels[mr];
            html += '<tr><td class="tm-left">' + WB.esc(rmName) + '</td>';
            for (var wk = 0; wk < pwWells.length; wk++) {
              var wName = pwWells[wk];
              var cellVal = pwLookup[rmName] ? pwLookup[rmName][wName] : null;
              var col = greenToRedColor(cellVal, pwMin, pwMax);
              var tip = rmName + ' | ' + wName + ': ' + (isFiniteNum(cellVal) ? WB.fmt(cellVal, 1) + '%' : '—');
              html += '<td class="tm-heat-cell" style="background:' + col.bg + ';color:' + col.fg + ';" title="' + WB.esc(tip) + '">' +
                WB.esc(WB.fmt(cellVal, 1)) + '</td>';
            }
            html += '</tr>';
          }
          html += '</tbody></table></div>';
        }
        html += '</div>';
      }

      // 7. Card "Daily comparison for <current well>"
      var curWell = mState.currentWell || WB.state.currentWell || '—';
      html += '<div class="tm-card">';
      html += '<div class="tm-card-header">';
      html += '<h3 class="tm-title">Daily comparison for ' + WB.esc(curWell) + '</h3>';
      html += '<div class="tm-row">';
      html += '<label class="tm-muted" for="tm-daily-phase">Phase:</label>';
      html += '<select id="tm-daily-phase" class="tm-select">';
      var phases = ['oil', 'water', 'liquid', 'gas'];
      for (var ph = 0; ph < phases.length; ph++) {
        html += '<option value="' + phases[ph] + '"' + (mState.dailyPhase === phases[ph] ? ' selected' : '') + '>' + phases[ph] + '</option>';
      }
      html += '</select>';
      html += '<label class="tm-check-item"><input type="checkbox" id="tm-unanchored-chk"' + (mState.unanchored ? ' checked' : '') + '> Unanchored (VFM)</label>';
      html += '<button id="tm-compute-btn" class="tm-btn tm-btn-primary">Compute selected</button>';
      html += '</div>';
      html += '</div>';

      // Model checkboxes
      html += '<div class="tm-checks" id="tm-model-checks">';
      var checkedMap = mState.checkedModels || {};
      for (var ci = 0; ci < modelsList.length; ci++) {
        var mod = modelsList[ci];
        html += '<label class="tm-check-item" title="' + WB.esc(mod.description || '') + '">' +
          '<input type="checkbox" class="tm-model-chk" value="' + WB.esc(mod.name) + '"' + (checkedMap[mod.name] ? ' checked' : '') + '> ' +
          '<span>' + WB.esc(mod.name) + '</span> ' + statusBadgeHtml(mod.status) +
          '</label>';
      }
      html += '</div>';

      if (mState.loadingDaily) {
        html += '<div class="tm-muted">Loading daily comparison...</div>';
      } else if (!mState.wellData || !mState.wellData.models || Object.keys(mState.wellData.models).length === 0) {
        html += '<div class="tm-empty">Select models above and click "Compute selected" to load daily series.</div>';
      } else {
        html += '<div class="tm-chart-box-tall"><canvas id="tm-daily-canvas"></canvas></div>';
        html += '<div class="tm-table-wrap" style="margin-top:12px;">';
        html += '<table class="tm-table"><thead><tr><th class="tm-left">Model</th><th>cum_oil_alloc_stb</th><th>Diff vs physics_wls (%)</th></tr></thead><tbody>';
        var wdModels = mState.wellData.models;
        var refObj = wdModels.physics_wls;
        var refCum = refObj && isFiniteNum(refObj.cum_oil_alloc_stb) ? refObj.cum_oil_alloc_stb : null;
        var wdKeys = Object.keys(wdModels);
        for (var wk2 = 0; wk2 < wdKeys.length; wk2++) {
          var k = wdKeys[wk2];
          var entry = wdModels[k] || {};
          var cum = entry.cum_oil_alloc_stb;
          var diffStr = '—';
          if (k === 'physics_wls' && isFiniteNum(cum)) {
            diffStr = '0.0%';
          } else if (isFiniteNum(cum) && isFiniteNum(refCum) && refCum !== 0) {
            var pct = ((cum - refCum) / refCum) * 100;
            diffStr = (pct >= 0 ? '+' : '') + WB.fmt(pct, 2) + '%';
          }
          html += '<tr><td class="tm-left">' + WB.esc(k) + '</td><td>' + WB.esc(WB.fmt(cum, 1)) + '</td><td>' + WB.esc(diffStr) + '</td></tr>';
        }
        html += '</tbody></table></div>';
      }

      html += '</div>'; // end daily card
      html += '</div>'; // end tm-root

      root.innerHTML = html;

      // Wire events
      var rerunBtn = document.getElementById('tm-rerun-bench');
      if (rerunBtn) {
        rerunBtn.addEventListener('click', function () {
          WB.post('/api/benchmark/run', {}, 'Starting benchmark...').then(function () {
            if (!mState.modelsData) mState.modelsData = {};
            mState.modelsData.benchmark_status = 'running';
            renderModelsTab();
            startBenchPoll();
          }).catch(function () {});
        });
      }

      var protoSeg = document.getElementById('tm-proto-seg');
      if (protoSeg) {
        protoSeg.addEventListener('click', function (e) {
          var btn = e.target.closest('button[data-proto]');
          if (!btn) return;
          var next = btn.getAttribute('data-proto');
          if (next && next !== mState.protocol) {
            mState.protocol = next;
            renderModelsTab();
          }
        });
      }

      var metricSel = document.getElementById('tm-metric-sel');
      if (metricSel) {
        metricSel.addEventListener('change', function () {
          mState.metric = metricSel.value;
          renderModelsTab();
        });
      }

      var sumTable = document.getElementById('tm-summary-table');
      if (sumTable) {
        sumTable.addEventListener('click', function (e) {
          var th = e.target.closest('th[data-sort]');
          if (th) {
            var sk = th.getAttribute('data-sort');
            if (mState.sortKey === sk) {
              mState.sortDir = mState.sortDir === 'asc' ? 'desc' : 'asc';
            } else {
              mState.sortKey = sk;
              mState.sortDir = 'asc';
            }
            renderModelsTab();
            return;
          }
          var tr = e.target.closest('tr[data-model]');
          if (tr) {
            mState.selectedModel = tr.getAttribute('data-model');
            renderModelsTab();
          }
        });
      }

      var phaseSel = document.getElementById('tm-daily-phase');
      if (phaseSel) {
        phaseSel.addEventListener('change', function () {
          mState.dailyPhase = phaseSel.value;
          renderModelsTab();
        });
      }

      var unanchoredChk = document.getElementById('tm-unanchored-chk');
      if (unanchoredChk) {
        unanchoredChk.addEventListener('change', function () {
          mState.unanchored = unanchoredChk.checked;
          renderModelsTab();
        });
      }

      var modelChecks = document.getElementById('tm-model-checks');
      if (modelChecks) {
        modelChecks.addEventListener('change', function (e) {
          if (e.target && e.target.classList.contains('tm-model-chk')) {
            if (!mState.checkedModels) mState.checkedModels = {};
            mState.checkedModels[e.target.value] = e.target.checked;
          }
        });
      }

      var computeBtn = document.getElementById('tm-compute-btn');
      if (computeBtn) {
        computeBtn.addEventListener('click', function () {
          var names = getCheckedModelNames();
          if (names.length === 0) {
            WB.toast('Select at least one model.', 'warn');
            return;
          }
          WB.post('/api/models/compute', { names: names }, 'Computing selected models...').then(function () {
            return WB.api('/api/models');
          }).then(function (md) {
            mState.modelsData = md;
            WB.emit('models', md);
            var map = getModelsMap();
            var allDone = true;
            for (var i = 0; i < names.length; i++) {
              var info = map[names[i]];
              if (!info || !isTerminalStatus(info.status)) {
                allDone = false;
                break;
              }
            }
            if (allDone) {
              loadDailyComparison(false);
            } else {
              renderModelsTab();
              startModelsPoll(names, function () {
                loadDailyComparison(false);
              });
            }
          }).catch(function () {});
        });
      }

      // Render Charts
      if (bd.available && summaryRows.length > 0) {
        renderBenchmarkBarChart(summaryRows);
        renderBenchmarkScatterChart();
      }
      if (mState.wellData && mState.wellData.models && Object.keys(mState.wellData.models).length > 0) {
        renderDailyChart();
      }
    }

    function renderBenchmarkBarChart(summaryRows) {
      var canvas = document.getElementById('tm-bar-canvas');
      if (!canvas) return;
      var items = [];
      for (var i = 0; i < summaryRows.length; i++) {
        var r = summaryRows[i];
        var v = r[mState.metric];
        if (isFiniteNum(v)) {
          items.push({ model: r.model || r.name || '', value: v });
        }
      }
      items.sort(function (a, b) { return a.value - b.value; });

      var palette = WB.palette || ['#2563eb'];
      var labels = [];
      var values = [];
      var colors = [];
      for (var j = 0; j < items.length; j++) {
        labels.push(items[j].model);
        values.push(items[j].value);
        colors.push(items[j].model === mState.selectedModel ? '#2563eb' : palette[j % palette.length]);
      }

      WB.chart(canvas, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: mState.metric,
            data: values,
            backgroundColor: colors,
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            x: { beginAtZero: true }
          }
        }
      });
    }

    function renderBenchmarkScatterChart() {
      var canvas = document.getElementById('tm-bench-scatter-canvas');
      if (!canvas) return;
      var pts = getFilteredPoints(mState.selectedModel);
      var byWell = {};
      var wellOrder = [];
      var minV = Infinity;
      var maxV = -Infinity;

      for (var i = 0; i < pts.length; i++) {
        var xy = extractPointXY(pts[i]);
        if (!isFiniteNum(xy.x) || !isFiniteNum(xy.y)) continue;
        if (!byWell[xy.well]) {
          byWell[xy.well] = [];
          wellOrder.push(xy.well);
        }
        byWell[xy.well].push({ x: xy.x, y: xy.y });
        if (xy.x < minV) minV = xy.x;
        if (xy.y < minV) minV = xy.y;
        if (xy.x > maxV) maxV = xy.x;
        if (xy.y > maxV) maxV = xy.y;
      }

      wellOrder.sort();
      var palette = WB.palette || ['#2563eb'];
      var datasets = [];
      for (var w = 0; w < wellOrder.length; w++) {
        var wn = wellOrder[w];
        datasets.push({
          label: wn,
          data: byWell[wn],
          backgroundColor: palette[w % palette.length],
          pointRadius: 4
        });
      }

      if (isFiniteNum(minV) && isFiniteNum(maxV) && maxV >= minV) {
        datasets.push({
          type: 'line',
          label: 'y = x',
          data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
          borderColor: '#6b7280',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        });
      }

      WB.chart(canvas, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'True liquid (STB/d)' } },
            y: { title: { display: true, text: 'Predicted liquid (STB/d)' } }
          }
        }
      });
    }

    function mapTestDateToLabel(tMid, dates) {
      if (!tMid || !dates || dates.length === 0) return null;
      var iso = String(tMid).slice(0, 10);
      for (var i = 0; i < dates.length; i++) {
        if (dates[i] === iso) return dates[i];
      }
      var targetMs = Date.parse(iso);
      if (!isFiniteNum(targetMs)) return null;
      var bestDate = null;
      var bestDiff = Infinity;
      for (var j = 0; j < dates.length; j++) {
        var dMs = Date.parse(dates[j]);
        if (!isFiniteNum(dMs)) continue;
        var diff = Math.abs(dMs - targetMs);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestDate = dates[j];
        }
      }
      return bestDate;
    }

    function renderDailyChart() {
      var canvas = document.getElementById('tm-daily-canvas');
      if (!canvas || !mState.wellData) return;

      var wd = mState.wellData;
      var dates = wd.dates || [];
      var models = wd.models || {};
      var phase = mState.dailyPhase || 'liquid';
      var suffix = mState.unanchored ? '_vfm' : '_alloc';
      var field = phase + suffix;
      var unit = WB.phaseUnit(phase);
      var palette = WB.palette || ['#2563eb'];

      var datasets = [];
      var modelNames = Object.keys(models);
      for (var i = 0; i < modelNames.length; i++) {
        var mName = modelNames[i];
        var mObj = models[mName] || {};
        var arr = mObj[field];
        if (!Array.isArray(arr)) continue;
        var col = palette[i % palette.length];
        datasets.push({
          label: mName + (mState.unanchored ? ' (VFM)' : ' (alloc)'),
          data: arr,
          borderColor: col,
          backgroundColor: col,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          spanGaps: true
        });
      }

      var tests = wd.tests || [];
      var testPts = [];
      for (var t = 0; t < tests.length; t++) {
        var tst = tests[t];
        var yVal = tst[phase];
        var lbl = mapTestDateToLabel(tst.t_mid, dates);
        if (lbl && isFiniteNum(yVal)) {
          testPts.push({ x: lbl, y: yVal });
        }
      }
      if (testPts.length > 0) {
        datasets.push({
          label: 'Well tests',
          data: testPts,
          borderColor: '#111827',
          backgroundColor: '#111827',
          showLine: false,
          pointRadius: 5,
          pointHoverRadius: 7
        });
      }

      WB.chart(canvas, {
        type: 'line',
        data: {
          labels: dates,
          datasets: datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            y: {
              title: { display: true, text: phase + ' (' + unit + ')' }
            }
          }
        }
      });
    }

    function handleModelsWellChange(well) {
      if (!well || well === mState.currentWell) return;
      mState.currentWell = well;
      if (mState.rendered && mState.modelsData) {
        loadDailyComparison(true);
      }
    }

    WB.on('well', function (well) {
      handleModelsWellChange(well);
    });

    WB.on('models', function (md) {
      if (md && md !== mState.modelsData) {
        mState.modelsData = md;
        if (mState.rendered) renderModelsTab();
      }
    });

    WB.registerTab({
      id: 'models',
      title: 'Model comparison',
      order: 4,
      render: function (container) {
        mState.container = container;
        mState.rendered = true;
        mState.currentWell = WB.state.currentWell || null;
        container.innerHTML = '<div class="tm-card"><div class="tm-muted">Loading model comparison...</div></div>';
      },
      onShow: function () {
        loadModelsTabData(true);
      },
      onWellChange: function (well) {
        handleModelsWellChange(well);
      }
    });

    // =========================================================================
    // TAB 5: CORRELATIONS
    // =========================================================================
    var cState = {
      container: null,
      rendered: false,
      source: 'well_tests',
      well: 'all',
      method: 'pearson',
      filter: '',
      threshold: 0,
      corrData: null,
      scatterData: null,
      selectedPair: null, // { x, y }
      loadingCorr: false,
      loadingScatter: false
    };

    var SOURCE_META = [
      { value: 'well_tests', label: 'well_tests', desc: 'well tests: 85 measured tests' },
      { value: 'daily', label: 'daily', desc: 'daily: active allocation model + daily sensor means' },
      { value: 'hourly', label: 'hourly', desc: 'hourly: hourly sensor means' }
    ];

    function getFilteredCorrIndices() {
      var cd = cState.corrData;
      var vars = (cd && cd.variables) || [];
      var q = cState.filter.trim().toLowerCase();
      var idxs = [];
      for (var i = 0; i < vars.length; i++) {
        if (!q || vars[i].toLowerCase().indexOf(q) >= 0) {
          idxs.push(i);
        }
      }
      return idxs;
    }

    function computeStrongestPairs(idxs) {
      var cd = cState.corrData;
      if (!cd || !cd.variables || !cd.matrix) return [];
      var vars = cd.variables;
      var mat = cd.matrix;
      var pairs = [];
      for (var a = 0; a < idxs.length; a++) {
        var i = idxs[a];
        var row = mat[i] || [];
        for (var b = a + 1; b < idxs.length; b++) {
          var j = idxs[b];
          var r = row[j];
          if (!isFiniteNum(r)) continue;
          pairs.push({
            x: vars[i],
            y: vars[j],
            r: r,
            absR: Math.abs(r)
          });
        }
      }
      pairs.sort(function (p1, p2) { return p2.absR - p1.absR; });
      return pairs.slice(0, 15);
    }

    function loadScatterPair(x, y) {
      if (!x || !y) {
        cState.scatterData = null;
        renderCorrTab();
        return;
      }
      cState.selectedPair = { x: x, y: y };
      cState.loadingScatter = true;
      renderCorrTab();
      var url = '/api/scatter?source=' + encodeURIComponent(cState.source) +
        '&well=' + encodeURIComponent(cState.well) +
        '&x=' + encodeURIComponent(x) +
        '&y=' + encodeURIComponent(y);
      WB.api(url).then(function (data) {
        cState.loadingScatter = false;
        cState.scatterData = data;
        renderCorrTab();
      }).catch(function (err) {
        cState.loadingScatter = false;
        WB.toast('Failed to load scatter: ' + err.message, 'error');
        renderCorrTab();
      });
    }

    function loadCorrData() {
      cState.loadingCorr = true;
      renderCorrTab();
      var url = '/api/corr?source=' + encodeURIComponent(cState.source) +
        '&well=' + encodeURIComponent(cState.well) +
        '&method=' + encodeURIComponent(cState.method);
      WB.api(url).then(function (data) {
        cState.loadingCorr = false;
        cState.corrData = data;

        var vars = data.variables || [];
        var hasX = cState.selectedPair && vars.indexOf(cState.selectedPair.x) >= 0;
        var hasY = cState.selectedPair && vars.indexOf(cState.selectedPair.y) >= 0;
        if (hasX && hasY) {
          loadScatterPair(cState.selectedPair.x, cState.selectedPair.y);
        } else {
          var allIdx = [];
          for (var i = 0; i < vars.length; i++) allIdx.push(i);
          var top = computeStrongestPairs(allIdx);
          if (top.length > 0) {
            loadScatterPair(top[0].x, top[0].y);
          } else if (vars.length >= 2) {
            loadScatterPair(vars[0], vars[1]);
          } else {
            cState.selectedPair = null;
            cState.scatterData = null;
            renderCorrTab();
          }
        }
      }).catch(function (err) {
        cState.loadingCorr = false;
        WB.toast('Failed to load correlations: ' + err.message, 'error');
        renderCorrTab();
      });
    }

    function buildWellOptions() {
      var seen = { all: true };
      var list = ['all'];
      var ovWells = getOverviewWells();
      for (var i = 0; i < ovWells.length; i++) {
        if (!seen[ovWells[i]]) { seen[ovWells[i]] = true; list.push(ovWells[i]); }
      }
      var cdWells = (cState.corrData && cState.corrData.wells) || [];
      for (var j = 0; j < cdWells.length; j++) {
        if (!seen[cdWells[j]]) { seen[cdWells[j]] = true; list.push(cdWells[j]); }
      }
      if (cState.well && !seen[cState.well]) {
        list.push(cState.well);
      }
      return list;
    }

    function renderCorrTab() {
      var root = cState.container;
      if (!root) return;

      var activeInputId = document.activeElement ? document.activeElement.id : null;
      var selStart = (activeInputId === 'tm-corr-filter' && document.activeElement.selectionStart !== undefined)
        ? document.activeElement.selectionStart : null;

      var cd = cState.corrData;
      var wellOpts = buildWellOptions();
      var html = '<div class="tm-root">';

      // 1. Controls card
      html += '<div class="tm-card">';
      html += '<div class="tm-row" style="justify-content:space-between;">';
      html += '<div class="tm-row">';

      html += '<label class="tm-muted" for="tm-corr-source">Source:</label>';
      html += '<select id="tm-corr-source" class="tm-select">';
      for (var sm = 0; sm < SOURCE_META.length; sm++) {
        var sMeta = SOURCE_META[sm];
        html += '<option value="' + sMeta.value + '"' + (cState.source === sMeta.value ? ' selected' : '') + '>' + WB.esc(sMeta.label) + '</option>';
      }
      html += '</select>';

      html += '<label class="tm-muted" for="tm-corr-well">Well:</label>';
      html += '<select id="tm-corr-well" class="tm-select">';
      for (var wo = 0; wo < wellOpts.length; wo++) {
        var wOpt = wellOpts[wo];
        html += '<option value="' + WB.esc(wOpt) + '"' + (cState.well === wOpt ? ' selected' : '') + '>' + WB.esc(wOpt) + '</option>';
      }
      html += '</select>';

      html += '<label class="tm-muted" for="tm-corr-method">Method:</label>';
      html += '<select id="tm-corr-method" class="tm-select">';
      var methods = ['pearson', 'spearman', 'kendall'];
      for (var me = 0; me < methods.length; me++) {
        html += '<option value="' + methods[me] + '"' + (cState.method === methods[me] ? ' selected' : '') + '>' + methods[me] + '</option>';
      }
      html += '</select>';

      html += '<label class="tm-muted" for="tm-corr-filter">Filter:</label>';
      html += '<input type="text" id="tm-corr-filter" class="tm-input" placeholder="variable substring..." value="' + WB.esc(cState.filter) + '">';

      html += '<label class="tm-muted" for="tm-corr-thresh">|r| &ge; <span id="tm-corr-thresh-val">' + WB.esc(cState.threshold.toFixed(1)) + '</span></label>';
      html += '<input type="range" id="tm-corr-thresh" min="0" max="0.9" step="0.1" value="' + WB.esc(String(cState.threshold)) + '">';

      html += '</div>';
      html += '<div class="tm-muted">' + (cd && isFiniteNum(cd.n_rows) ? 'n_rows: ' + WB.esc(String(cd.n_rows)) : '') + '</div>';
      html += '</div>';

      // One-line explanations of each source
      html += '<div class="tm-muted" style="margin-top:8px;">';
      var descs = [];
      for (var d = 0; d < SOURCE_META.length; d++) {
        var item = SOURCE_META[d];
        if (item.value === cState.source) {
          descs.push('<strong>' + WB.esc(item.desc) + '</strong>');
        } else {
          descs.push(WB.esc(item.desc));
        }
      }
      html += descs.join(' &middot; ');
      html += '</div>';
      html += '</div>'; // end controls card

      if (cState.loadingCorr && !cd) {
        html += '<div class="tm-card"><div class="tm-muted">Loading correlation matrix...</div></div>';
      } else if (!cd || !cd.variables || cd.variables.length === 0) {
        html += '<div class="tm-card"><div class="tm-empty">No correlation data available.</div></div>';
      } else {
        var idxs = getFilteredCorrIndices();
        var vars = cd.variables;
        var mat = cd.matrix || [];
        var showNums = idxs.length <= 20;
        var topPairs = computeStrongestPairs(idxs);

        // 2. Heatmap card
        html += '<div class="tm-card">';
        html += '<div class="tm-card-header"><h3 class="tm-title">Correlation matrix (' + WB.esc(String(idxs.length)) + ' variables)</h3><span class="tm-muted">Click any cell to inspect scatter</span></div>';
        if (idxs.length === 0) {
          html += '<div class="tm-empty">No variables match filter "' + WB.esc(cState.filter) + '".</div>';
        } else {
          html += '<div class="tm-corr-wrap"><table class="tm-corr-table" id="tm-corr-table">';
          html += '<thead><tr><th class="tm-corr-corner"></th>';
          for (var c = 0; c < idxs.length; c++) {
            var cName = vars[idxs[c]];
            html += '<th class="tm-corr-col-th" title="' + WB.esc(cName) + '"><span class="tm-corr-col-lbl">' + WB.esc(cName) + '</span></th>';
          }
          html += '</tr></thead><tbody>';

          for (var rIdx = 0; rIdx < idxs.length; rIdx++) {
            var i = idxs[rIdx];
            var rName = vars[i];
            var rowArr = mat[i] || [];
            html += '<tr><th class="tm-corr-row-th">' + WB.esc(rName) + '</th>';
            for (var cIdx = 0; cIdx < idxs.length; cIdx++) {
              var j = idxs[cIdx];
              var cVar = vars[j];
              var val = rowArr[j];
              var col = divergingColor(val);
              var isDim = isFiniteNum(val) && Math.abs(val) < cState.threshold;
              var isSel = cState.selectedPair &&
                ((cState.selectedPair.x === rName && cState.selectedPair.y === cVar) ||
                 (cState.selectedPair.x === cVar && cState.selectedPair.y === rName));
              var tip = rName + ' vs ' + cVar + ': ' + (isFiniteNum(val) ? val.toFixed(3) : 'null');
              var cellText = (showNums && isFiniteNum(val)) ? val.toFixed(2) : '';
              html += '<td class="tm-corr-cell' + (isDim ? ' tm-dim' : '') + (isSel ? ' tm-corr-sel' : '') + '" ' +
                'data-x="' + WB.esc(rName) + '" data-y="' + WB.esc(cVar) + '" ' +
                'style="background:' + col.bg + ';color:' + col.fg + ';" ' +
                'title="' + WB.esc(tip) + '">' + WB.esc(cellText) + '</td>';
            }
            html += '</tr>';
          }
          html += '</tbody></table></div>';
        }
        html += '</div>';

        // 3 & 4. Scatter + Strongest pairs
        html += '<div class="tm-grid-2">';

        html += '<div class="tm-card">';
        var sd = cState.scatterData;
        var pairTitle = cState.selectedPair
          ? (cState.selectedPair.x + ' vs ' + cState.selectedPair.y)
          : 'Selected pair';
        var pairStats = sd
          ? ('r = ' + WB.fmt(sd.pearson_r, 3) + ', n = ' + (isFiniteNum(sd.n) ? sd.n : '—'))
          : (cState.loadingScatter ? 'Loading...' : '');
        html += '<div class="tm-card-header"><h3 class="tm-title">' + WB.esc(pairTitle) + '</h3><span class="tm-muted">' + WB.esc(pairStats) + '</span></div>';
        if (!cState.selectedPair) {
          html += '<div class="tm-empty">Click a cell in the heatmap or a pair in Strongest pairs.</div>';
        } else {
          html += '<div class="tm-chart-box-tall"><canvas id="tm-corr-scatter-canvas"></canvas></div>';
        }
        html += '</div>';

        html += '<div class="tm-card">';
        html += '<div class="tm-card-header"><h3 class="tm-title">Strongest pairs (top 15 |r|)</h3><span class="tm-muted">Click to plot</span></div>';
        if (topPairs.length === 0) {
          html += '<div class="tm-empty">Not enough variables for pairs.</div>';
        } else {
          html += '<ul class="tm-pair-list" id="tm-pair-list">';
          for (var tp = 0; tp < topPairs.length; tp++) {
            var p = topPairs[tp];
            var pSel = cState.selectedPair &&
              ((cState.selectedPair.x === p.x && cState.selectedPair.y === p.y) ||
               (cState.selectedPair.x === p.y && cState.selectedPair.y === p.x));
            html += '<li class="tm-pair-item' + (pSel ? ' tm-pair-sel' : '') + '" data-x="' + WB.esc(p.x) + '" data-y="' + WB.esc(p.y) + '">' +
              '<span>' + WB.esc(p.x) + ' vs ' + WB.esc(p.y) + '</span>' +
              '<strong>' + WB.esc(p.r.toFixed(3)) + '</strong>' +
              '</li>';
          }
          html += '</ul>';
        }
        html += '</div>';

        html += '</div>';
      }

      html += '</div>';
      root.innerHTML = html;

      // Wire events
      var srcSel = document.getElementById('tm-corr-source');
      if (srcSel) {
        srcSel.addEventListener('change', function () {
          cState.source = srcSel.value;
          loadCorrData();
        });
      }

      var wellSel = document.getElementById('tm-corr-well');
      if (wellSel) {
        wellSel.addEventListener('change', function () {
          cState.well = wellSel.value;
          loadCorrData();
        });
      }

      var methSel = document.getElementById('tm-corr-method');
      if (methSel) {
        methSel.addEventListener('change', function () {
          cState.method = methSel.value;
          loadCorrData();
        });
      }

      var filterInput = document.getElementById('tm-corr-filter');
      if (filterInput) {
        filterInput.addEventListener('input', function () {
          cState.filter = filterInput.value;
          renderCorrTab();
        });
        if (activeInputId === 'tm-corr-filter') {
          filterInput.focus();
          if (selStart !== null) {
            filterInput.setSelectionRange(selStart, selStart);
          }
        }
      }

      var threshSlider = document.getElementById('tm-corr-thresh');
      if (threshSlider) {
        threshSlider.addEventListener('input', function () {
          cState.threshold = parseFloat(threshSlider.value) || 0;
          var lbl = document.getElementById('tm-corr-thresh-val');
          if (lbl) lbl.textContent = cState.threshold.toFixed(1);
          var cells = root.querySelectorAll('.tm-corr-cell');
          var cdNow = cState.corrData;
          if (!cdNow || !cdNow.variables || !cdNow.matrix) return;
          var varIndex = {};
          for (var vi = 0; vi < cdNow.variables.length; vi++) {
            varIndex[cdNow.variables[vi]] = vi;
          }
          for (var ci = 0; ci < cells.length; ci++) {
            var cell = cells[ci];
            var xi = varIndex[cell.getAttribute('data-x')];
            var yi = varIndex[cell.getAttribute('data-y')];
            var rv = (xi !== undefined && yi !== undefined) ? cdNow.matrix[xi][yi] : null;
            if (isFiniteNum(rv) && Math.abs(rv) < cState.threshold) {
              cell.classList.add('tm-dim');
            } else {
              cell.classList.remove('tm-dim');
            }
          }
        });
      }

      var corrTable = document.getElementById('tm-corr-table');
      if (corrTable) {
        corrTable.addEventListener('click', function (e) {
          var td = e.target.closest('td.tm-corr-cell');
          if (!td) return;
          var x = td.getAttribute('data-x');
          var y = td.getAttribute('data-y');
          if (x && y) loadScatterPair(x, y);
        });
      }

      var pairList = document.getElementById('tm-pair-list');
      if (pairList) {
        pairList.addEventListener('click', function (e) {
          var li = e.target.closest('li.tm-pair-item');
          if (!li) return;
          var x = li.getAttribute('data-x');
          var y = li.getAttribute('data-y');
          if (x && y) loadScatterPair(x, y);
        });
      }

      if (cState.selectedPair && cState.scatterData) {
        renderCorrScatterChart();
      }
    }

    function renderCorrScatterChart() {
      var canvas = document.getElementById('tm-corr-scatter-canvas');
      var sd = cState.scatterData;
      if (!canvas || !sd) return;

      var pts = sd.points || [];
      var byWell = {};
      var wellOrder = [];
      for (var i = 0; i < pts.length; i++) {
        var p = pts[i];
        if (!isFiniteNum(p.x) || !isFiniteNum(p.y)) continue;
        var wn = p.well_name || 'Unknown';
        if (!byWell[wn]) {
          byWell[wn] = [];
          wellOrder.push(wn);
        }
        byWell[wn].push({ x: p.x, y: p.y });
      }
      wellOrder.sort();

      var palette = WB.palette || ['#2563eb'];
      var datasets = [];
      for (var w = 0; w < wellOrder.length; w++) {
        var name = wellOrder[w];
        datasets.push({
          label: name,
          data: byWell[name],
          backgroundColor: palette[w % palette.length],
          pointRadius: 4
        });
      }

      WB.chart(canvas, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: true, position: 'top' }
          },
          scales: {
            x: { title: { display: true, text: sd.x || (cState.selectedPair && cState.selectedPair.x) || 'x' } },
            y: { title: { display: true, text: sd.y || (cState.selectedPair && cState.selectedPair.y) || 'y' } }
          }
        }
      });
    }

    function handleCorrWellChange(well) {
      if (!well || cState.well === 'all' || cState.well === well) return;
      cState.well = well;
      if (cState.rendered) {
        loadCorrData();
      }
    }

    WB.on('well', function (well) {
      handleCorrWellChange(well);
    });

    WB.on('overview', function () {
      if (cState.rendered) renderCorrTab();
    });

    WB.registerTab({
      id: 'corr',
      title: 'Correlations',
      order: 5,
      render: function (container) {
        cState.container = container;
        cState.rendered = true;
        container.innerHTML = '<div class="tm-card"><div class="tm-muted">Loading correlations...</div></div>';
      },
      onShow: function () {
        if (!cState.corrData) {
          loadCorrData();
        } else {
          renderCorrTab();
        }
      },
      onWellChange: function (well) {
        handleCorrWellChange(well);
      }
    });
  }

  boot();
})();
