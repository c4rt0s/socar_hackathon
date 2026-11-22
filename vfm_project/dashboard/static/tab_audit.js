(function () {
  'use strict';

  function boot() {
    if (!window.WB || typeof window.WB.registerTab !== 'function') {
      setTimeout(boot, 25);
      return;
    }

    var WB = window.WB;
    var root = null;
    var items = []; // [{id, name, builtin, content_b64, score}]
    var refFile = null; // {name, content_b64}
    var lastResult = null;
    var seqCounter = 0;

    function readFileBase64(file) {
      return new Promise(function (resolve, reject) {
        var fr = new FileReader();
        fr.onload = function () { resolve({ name: file.name, content_b64: String(fr.result || '') }); };
        fr.onerror = function () { reject(new Error('Failed reading ' + file.name)); };
        fr.readAsDataURL(file);
      });
    }

    function addBuiltin(builtinKey, label, defaultScore) {
      for (var i = 0; i < items.length; i++) {
        if (items[i].builtin === builtinKey) return;
      }
      items.push({
        id: ++seqCounter,
        name: label,
        builtin: builtinKey,
        content_b64: null,
        score: defaultScore !== undefined ? defaultScore : ''
      });
      renderFileQueue();
    }

    function renderFileQueue() {
      if (!root) return;
      var body = root.querySelector('#aud-queue-body');
      if (!items.length) {
        body.innerHTML = '<tr><td colspan="4" class="small-muted">Upload CSV/Excel files or add built-in candidates above.</td></tr>';
        return;
      }
      body.innerHTML = items.map(function (it) {
        return '<tr>' +
          '<td><strong>' + WB.esc(it.name) + '</strong></td>' +
          '<td>' + (it.builtin ? '<span class="badge blue">Built-in (' + WB.esc(it.builtin) + ')</span>' : '<span class="badge">Uploaded</span>') + '</td>' +
          '<td><input type="number" step="0.1" min="0" placeholder="e.g. 20" data-score-id="' + it.id + '" value="' + WB.esc(it.score) + '" style="width:90px"></td>' +
          '<td><button type="button" class="danger" data-rm-id="' + it.id + '">Remove</button></td>' +
        '</tr>';
      }).join('');

      body.querySelectorAll('[data-score-id]').forEach(function (inp) {
        inp.addEventListener('input', function () {
          var id = Number(inp.getAttribute('data-score-id'));
          for (var i = 0; i < items.length; i++) {
            if (items[i].id === id) items[i].score = inp.value;
          }
        });
      });
      body.querySelectorAll('[data-rm-id]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var id = Number(btn.getAttribute('data-rm-id'));
          items = items.filter(function (x) { return x.id !== id; });
          renderFileQueue();
        });
      });
    }

    function renderResults(res) {
      if (!root || !res) return;
      var wrap = root.querySelector('#aud-results');
      wrap.style.display = 'flex';

      var evals = res.evals || [];
      var bc = res.backcalc || {};
      var maxS = bc.max_score || 30;

      // 1. Leaderboard + Diagnostics
      var lbHtml = evals.map(function (ev) {
        var m = ev.metrics || {};
        var o = m.oil || {}, w = m.water || {}, g = m.gas || {};
        var au = ev.audit || {};
        var si = ev.shutin_check || {};
        var flags = [];
        if (!au.date_column_exact) flags.push('Col="' + au.column_map.date + '" (expected "date")');
        if (au.n_nan_rows > 0) flags.push(au.n_nan_rows + ' NaNs');
        if (au.n_negative_rows > 0) flags.push(au.n_negative_rows + ' negatives');
        if (si.n_nonzero_shutin_days > 0) flags.push(si.n_nonzero_shutin_days + ' non-zero shut-ins');
        var gt = ev.gas_if_scored_against_total_gas;
        if (gt && g.mape != null && gt.mape != null && gt.mape + 1.0 < g.mape) {
          flags.push('Looks closer to Total Gas (' + WB.fmt(gt.mape, 2) + '% vs ' + WB.fmt(g.mape, 2) + '%)');
        }
        var upo = ev.oil_if_uptime_adjusted;
        if (upo && o.mape != null && upo.mape != null && upo.mape + 0.5 < o.mape) {
          flags.push('Flowing/uptime rate lowers Oil MAPE (' + WB.fmt(upo.mape, 2) + '% vs ' + WB.fmt(o.mape, 2) + '%)');
        }
        return '<tr>' +
          '<td><strong>' + WB.esc(ev.name) + '</strong></td>' +
          '<td class="num">' + (ev.reported_score != null ? WB.esc(WB.fmt(ev.reported_score, 2) + '/' + maxS) : '—') + '</td>' +
          '<td class="num">' + WB.esc(ev.n_matched + '/' + ev.n_ref + ' (' + ev.coverage_pct + '%)') + '</td>' +
          '<td class="num">' + WB.esc(WB.fmt(o.mape, 2)) + '%</td>' +
          '<td class="num">' + WB.esc(WB.fmt(w.mape, 2)) + '%</td>' +
          '<td class="num">' + WB.esc(WB.fmt(g.mape, 2)) + '%</td>' +
          '<td class="num">' + WB.esc(WB.fmt(o.mae, 1)) + ' / ' + WB.esc(WB.fmt(w.mae, 1)) + ' / ' + WB.esc(WB.fmt(g.mae, 1)) + '</td>' +
          '<td class="num">' + WB.esc(WB.fmt(ev.mean_owg_r2_pooled, 4)) + '</td>' +
          '<td class="num">' + WB.esc(WB.fmt(ev.mean_owg_r2_per_well, 3)) + '</td>' +
          '<td>' + (flags.length ? WB.esc(flags.join(' · ')) : '<span class="chip green">Clean</span>') + '</td>' +
        '</tr>';
      }).join('');
      root.querySelector('#aud-lb-body').innerHTML = lbHtml || '<tr><td colspan="10" class="small-muted">No matches found.</td></tr>';
      root.querySelector('#aud-ref-note').textContent = 'Reference: ' + (res.reference ? res.reference.source + ' (' + res.reference.n_rows + ' rows)' : '—');

      // 2. Implied Targets + Backcalculated Formulas
      var imp = bc.implied_targets || [];
      root.querySelector('#aud-implied').innerHTML = imp.length
        ? imp.map(function (r) {
            return '<div class="note-item INFO"><span class="note-meta">[' + WB.esc(r.name) + ' · ' + WB.esc(r.reported_score + '/' + maxS + ' = ' + r.score_pct + '%') + ']</span>' +
              'Implied test performance: R² ≈ ' + WB.esc(r.implied_r2) +
              ' · Linear MAPE ≈ ' + WB.esc(r.implied_mape_if_cap30) + '% (cap 30%) or ' +
              WB.esc(r.implied_mape_if_cap25) + '% (cap 25%) · Inverse MAPE(10%) ≈ ' + WB.esc(r.implied_mape_if_inv10) + '%' +
            '</div>';
          }).join('')
        : '<div class="small-muted">Enter your reported score (e.g. 20) next to one or more submissions to backcalculate the grader metric.</div>';

      var subNames = bc.submission_names || [];
      root.querySelector('#aud-f-head').innerHTML = '<tr><th>Candidate Formula</th>' +
        subNames.map(function (n) { return '<th class="num">' + WB.esc(n) + '</th>'; }).join('') +
        '<th class="num">MAE vs Rep</th><th class="num">Corr r</th></tr>';

      var formulas = bc.formulas || [];
      root.querySelector('#aud-f-body').innerHTML = formulas.map(function (f) {
        return '<tr><td><strong>' + WB.esc(f.formula) + '</strong></td>' +
          (f.predicted_scores || []).map(function (p) { return '<td class="num">' + WB.esc(WB.fmt(p, 2)) + '</td>'; }).join('') +
          '<td class="num">' + WB.esc(WB.fmt(f.mae_vs_reported, 2)) + '</td>' +
          '<td class="num">' + WB.esc(WB.fmt(f.corr, 3)) + '</td>' +
        '</tr>';
      }).join('');

      // 3. Per-well comparison chart (Oil MAPE% per well)
      var wellsSet = {};
      evals.forEach(function (ev) {
        (ev.per_well || []).forEach(function (pw) { wellsSet[pw.well_name] = true; });
      });
      var wellsList = Object.keys(wellsSet).sort();
      var chartCanvas = root.querySelector('#aud-well-chart');
      if (wellsList.length && evals.length) {
        WB.chart(chartCanvas, {
          type: 'bar',
          data: {
            labels: wellsList,
            datasets: evals.map(function (ev, idx) {
              var map = {};
              (ev.per_well || []).forEach(function (pw) { map[pw.well_name] = pw.oil_mape; });
              return {
                label: ev.name + ' (Oil MAPE%)',
                data: wellsList.map(function (w) { return map[w] != null ? map[w] : null; }),
                backgroundColor: WB.palette[idx % WB.palette.length]
              };
            })
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: { y: { title: { display: true, text: 'Oil MAPE (%)' }, beginAtZero: true } }
          }
        });
      }

      // 4. Per-well table for top submission
      var pwRows = [];
      evals.forEach(function (ev) {
        (ev.per_well || []).forEach(function (pw) {
          pwRows.push('<tr><td>' + WB.esc(ev.name) + '</td><td><strong>' + WB.esc(pw.well_name) + '</strong></td>' +
            '<td class="num">' + pw.n + '</td>' +
            '<td class="num">' + WB.esc(WB.fmt(pw.oil_mape, 2)) + '%</td>' +
            '<td class="num">' + WB.esc(WB.fmt(pw.water_mape, 2)) + '%</td>' +
            '<td class="num">' + WB.esc(WB.fmt(pw.gas_mape, 2)) + '%</td>' +
            '<td class="num">' + WB.esc(WB.fmt(pw.oil_bias_pct, 2)) + '%</td>' +
          '</tr>');
        });
      });
      root.querySelector('#aud-pw-body').innerHTML = pwRows.join('');

      // 5. Pairwise diffs
      var pw = res.pairwise || [];
      root.querySelector('#aud-pair-host').innerHTML = pw.length
        ? pw.map(function (p) {
            var ph = p.phases || {};
            var top = (p.top_diverging_wells || []).map(function (w) {
              return w.well_name + ' (Δoil ' + w.oil_mae_diff + ' STB/d, Δgas ' + w.gas_mae_diff + ' Mscf/d)';
            }).join(' · ');
            return '<div class="note-item INFO"><strong>' + WB.esc(p.file_a + ' vs ' + p.file_b) + '</strong> (' + p.n_common_rows + ' common rows): ' +
              'Oil MAE=' + WB.esc(WB.fmt(ph.oil && ph.oil.mae, 1)) + ' STB/d · Water MAE=' + WB.esc(WB.fmt(ph.water && ph.water.mae, 1)) +
              ' · Gas MAE=' + WB.esc(WB.fmt(ph.gas && ph.gas.mae, 1)) + ' Mscf/d' +
              (top ? '<br><span class="small-muted">Largest well diffs: ' + WB.esc(top) + '</span>' : '') +
            '</div>';
          }).join('')
        : '<div class="small-muted">Add at least 2 submissions to see pairwise differences.</div>';
    }

    function runAudit() {
      if (!items.length) {
        WB.toast('Add at least one CSV/Excel or built-in model first.', 'warn');
        return;
      }
      var target = root.querySelector('#aud-target').value;
      if (target === 'reference' && !refFile) {
        WB.toast('Upload a reference CSV/Excel or switch Reference to Well Tests.', 'warn');
        return;
      }
      var maxScore = Number(root.querySelector('#aud-max-score').value) || 30;
      WB.post('/api/audit', {
        eval_target: target,
        max_score: maxScore,
        reference: target === 'reference' ? refFile : null,
        files: items.map(function (it) {
          return {
            name: it.name,
            builtin: it.builtin,
            content_b64: it.content_b64,
            score: it.score
          };
        })
      }, 'Auditing submissions and backcalculating metrics...').then(function (res) {
        lastResult = res;
        renderResults(res);
      });
    }

    function buildSkeleton(el) {
      el.innerHTML = [
        '<div style="display:flex; flex-direction:column; gap:12px;">',
          '<div class="card">',
            '<div class="card-header">',
              '<h3 class="card-title">VFM Score Audit &amp; Grader Metric Backcalculator</h3>',
              '<span class="small-muted">Upload multiple CSV/Excel VFM files, compare against well tests or ground truth, and solve which metric fits your judge scores</span>',
            '</div>',
            '<div class="form-row">',
              '<div class="form-field"><label>Upload candidate CSV / Excel files</label>',
                '<input type="file" id="aud-file-in" multiple accept=".csv,.xlsx,.xls">',
              '</div>',
              '<div class="form-field"><label>Reference truth</label>',
                '<select id="aud-target">',
                  '<option value="tests_q3">Well Tests — Q3 Out-of-Sample (&ge; 2025-07-01)</option>',
                  '<option value="tests_all">Well Tests — All</option>',
                  '<option value="reference">Uploaded Ground-Truth CSV/Excel</option>',
                '</select>',
              '</div>',
              '<div class="form-field" id="aud-ref-wrap" style="display:none"><label>Reference file</label>',
                '<input type="file" id="aud-ref-in" accept=".csv,.xlsx,.xls">',
              '</div>',
              '<div class="form-field"><label>Max score</label>',
                '<input type="number" id="aud-max-score" value="30" min="1" style="width:75px">',
              '</div>',
            '</div>',
            '<div class="toolbar" style="margin-bottom:8px; flex-wrap:wrap;">',
              '<span class="small-muted">Quick-add built-in models:</span>',
              '<button type="button" id="aud-add-synth-v2" class="primary">+ Synth Stacked V2 (New)</button>',
              '<button type="button" id="aud-add-synth-gbm">+ Synth GBM Ensemble (New)</button>',
              '<button type="button" id="aud-add-pw-alloc">+ Physics Baseline (alloc)</button>',
              '<button type="button" id="aud-add-pw-vfm">+ Physics Baseline (unanchored VFM)</button>',
              '<button type="button" id="aud-add-ridge-alloc">+ Ridge (alloc)</button>',
              '<button type="button" id="aud-run-btn" class="primary" style="margin-left:auto;">Run Score Audit</button>',
            '</div>',
            '<div class="table-wrap">',
              '<table class="data-table">',
                '<thead><tr><th>Submission</th><th>Source</th><th>Reported Score</th><th>Action</th></tr></thead>',
                '<tbody id="aud-queue-body"></tbody>',
              '</table>',
            '</div>',
          '</div>',

          '<div id="aud-results" style="display:none; flex-direction:column; gap:12px;">',
            '<div class="card">',
              '<div class="card-header"><h3 class="card-title">Submission Accuracy Leaderboard &amp; Format Audit</h3><span id="aud-ref-note" class="small-muted"></span></div>',
              '<div class="table-wrap">',
                '<table class="data-table">',
                  '<thead><tr><th>File</th><th class="num">Reported</th><th class="num">Matched</th><th class="num">Oil MAPE</th><th class="num">Water MAPE</th><th class="num">Gas MAPE</th><th class="num">MAE (O/W/G)</th><th class="num">Mean R² (pool)</th><th class="num">Mean R² (well)</th><th>Audit Flags</th></tr></thead>',
                  '<tbody id="aud-lb-body"></tbody>',
                '</table>',
              '</div>',
            '</div>',

            '<div class="grid-2">',
              '<div class="card">',
                '<div class="card-header"><h3 class="card-title">Grader Metric Backcalculator</h3></div>',
                '<div id="aud-implied" class="note-list" style="margin-bottom:10px;"></div>',
                '<div class="table-wrap" style="max-height:320px; overflow:auto;">',
                  '<table class="data-table"><thead id="aud-f-head"></thead><tbody id="aud-f-body"></tbody></table>',
                '</div>',
              '</div>',
              '<div class="card">',
                '<div class="card-header"><h3 class="card-title">Per-Well Oil Error (MAPE %)</h3></div>',
                '<div class="chart-box" style="height:260px"><canvas id="aud-well-chart"></canvas></div>',
                '<div id="aud-pair-host" class="note-list" style="margin-top:10px;"></div>',
              '</div>',
            '</div>',

            '<div class="card">',
              '<div class="card-header"><h3 class="card-title">Per-Well Breakdown</h3></div>',
              '<div class="table-wrap" style="max-height:300px; overflow:auto;">',
                '<table class="data-table">',
                  '<thead><tr><th>Submission</th><th>Well</th><th class="num">N</th><th class="num">Oil MAPE</th><th class="num">Water MAPE</th><th class="num">Gas MAPE</th><th class="num">Oil Bias%</th></tr></thead>',
                  '<tbody id="aud-pw-body"></tbody>',
                '</table>',
              '</div>',
            '</div>',
          '</div>',
        '</div>'
      ].join('');

      el.querySelector('#aud-target').addEventListener('change', function (e) {
        el.querySelector('#aud-ref-wrap').style.display = e.target.value === 'reference' ? '' : 'none';
      });
      el.querySelector('#aud-ref-in').addEventListener('change', function (e) {
        var f = e.target.files && e.target.files[0];
        if (!f) return;
        readFileBase64(f).then(function (obj) {
          refFile = obj;
          WB.toast('Loaded reference: ' + obj.name, 'info');
        });
      });
      el.querySelector('#aud-file-in').addEventListener('change', function (e) {
        var files = Array.prototype.slice.call(e.target.files || []);
        Promise.all(files.map(readFileBase64)).then(function (loaded) {
          loaded.forEach(function (f) {
            items.push({ id: ++seqCounter, name: f.name, builtin: null, content_b64: f.content_b64, score: '' });
          });
          renderFileQueue();
          e.target.value = '';
        });
      });

      el.querySelector('#aud-add-synth-v2').addEventListener('click', function () { addBuiltin('synth_stacked_v2:alloc', 'synth_stacked_v2', ''); });
      el.querySelector('#aud-add-synth-gbm').addEventListener('click', function () { addBuiltin('synth_gbm_ensemble:alloc', 'synth_gbm_ensemble', ''); });
      el.querySelector('#aud-add-pw-alloc').addEventListener('click', function () { addBuiltin('physics_wls:alloc', 'physics_wls_alloc', ''); });
      el.querySelector('#aud-add-pw-vfm').addEventListener('click', function () { addBuiltin('physics_wls:vfm', 'physics_wls_unanchored_vfm', ''); });
      el.querySelector('#aud-add-ridge-alloc').addEventListener('click', function () { addBuiltin('ridge:alloc', 'ridge_alloc', ''); });
      el.querySelector('#aud-run-btn').addEventListener('click', runAudit);

      addBuiltin('synth_stacked_v2:alloc', 'synth_stacked_v2', '');
      addBuiltin('synth_gbm_ensemble:alloc', 'synth_gbm_ensemble', '');
      addBuiltin('physics_wls:alloc', 'physics_wls_alloc', '20');
      addBuiltin('physics_wls:vfm', 'physics_wls_unanchored_vfm', '');
    }

    WB.registerTab({
      id: 'audit',
      title: 'Score Audit',
      order: 7,
      render: function (container) {
        root = container;
        buildSkeleton(container);
      },
      onShow: function () {
        if (!lastResult && items.length) renderFileQueue();
      }
    });
  }

  boot();
})();
