/**
 * Plotly / Spotfire / KAPPA Saphir-grade Interactive Chart Engine for Chart.js v4.
 *
 * Clear, unambiguous separation between SHIFTING (panning) and RESCALING (zooming):
 *   - Left Y / Right Y axis (middle): Drag UP/DOWN = SHIFT that axis independently;
 *                                     Scroll wheel = RESCALE that axis independently!
 *   - Left Y / Right Y axis (top/bottom edges): Drag OR Scroll = RESCALE that axis!
 *   - Bottom X axis (middle): Drag LEFT/RIGHT = SHIFT X; Scroll wheel = RESCALE X!
 *   - Bottom X axis (left/right edges): Drag OR Scroll = RESCALE X!
 *   - Main Plot Area:
 *       * Box Zoom mode ('box'): Drag draws a crisp zoom rectangle -> instant zoom!
 *       * Pan mode ('pan'):      Drag smoothly shifts X & Y 1:1 at 60 FPS!
 *       * Scroll wheel:          Smoothly zooms around cursor in BOTH modes!
 *   - Double-click any axis -> auto-fit ONLY that axis; Double-click plot -> reset all!
 */
(function () {
  'use strict';

  function isLog(scale) {
    return Boolean(scale && scale.type === 'logarithmic');
  }

  function scaleRange(min0, max0, anchor, factor, logMode, limits) {
    var lo, hi;
    if (logMode) {
      var lMin = Math.log10(Math.max(1e-6, min0));
      var lMax = Math.log10(Math.max(1e-5, max0));
      var lAnc = Math.log10(Math.max(1e-6, anchor));
      var nL0 = lAnc - (lAnc - lMin) * factor;
      var nL1 = lAnc + (lMax - lAnc) * factor;
      if (nL1 - nL0 < 0.08 || nL1 - nL0 > 9.0) return null;
      lo = Math.pow(10, Math.max(-5, nL0));
      hi = Math.pow(10, Math.min(8, nL1));
    } else {
      lo = anchor - (anchor - min0) * factor;
      hi = anchor + (max0 - anchor) * factor;
      if (!isFinite(lo) || !isFinite(hi) || hi - lo < 1e-6) return null;
      if (limits && limits.minSpan && (hi - lo) < limits.minSpan) return null;
      if (limits && limits.minVal !== undefined && lo < limits.minVal) lo = limits.minVal;
      if (limits && limits.maxVal !== undefined && hi > limits.maxVal) hi = limits.maxVal;
      if (limits && limits.minSpan && (hi - lo) < limits.minSpan) return null;
    }
    return [lo, hi];
  }

  function shiftLinearOrLog(min0, max0, fracDelta, logMode, limits) {
    if (logMode) {
      var l0 = Math.log10(Math.max(1e-6, min0));
      var l1 = Math.log10(Math.max(1e-5, max0));
      var sh = -fracDelta * (l1 - l0);
      var n0 = Math.max(-5, l0 + sh);
      var n1 = Math.min(8, l1 + sh);
      if (n1 - n0 < 0.05) return null;
      return [Math.pow(10, n0), Math.pow(10, n1)];
    }
    var span = max0 - min0;
    var lo = min0 - fracDelta * span;
    var hi = max0 - fracDelta * span;
    if (limits && limits.minVal !== undefined && limits.maxVal !== undefined) {
      if (lo < limits.minVal) { lo = limits.minVal; hi = lo + span; }
      if (hi > limits.maxVal) { hi = limits.maxVal; lo = Math.max(limits.minVal, hi - span); }
    }
    return [lo, hi];
  }

  function detectZone(ch, x, y, opts) {
    var a = ch.chartArea;
    if (!a) return 'none';
    var rightId = opts.rightScaleId;
    var hasRight = Boolean(rightId && ch.scales[rightId] && ch.scales[rightId].options && ch.scales[rightId].options.display !== false);
    var h = Math.max(1, a.bottom - a.top);
    var w = Math.max(1, a.right - a.left);

    // Generous hit targets on Left Y, Right Y, and Bottom X axes
    if (x >= 0 && x < a.left + 4 && y >= a.top - 10 && y <= a.bottom + 10) {
      var yfL = (y - a.top) / h;
      return (yfL < 0.18 || yfL > 0.82) ? 'y-left-edge' : 'y-left-mid';
    }
    if (hasRight && x > a.right - 4 && x <= ch.width && y >= a.top - 10 && y <= a.bottom + 10) {
      var yfR = (y - a.top) / h;
      return (yfR < 0.18 || yfR > 0.82) ? 'y-right-edge' : 'y-right-mid';
    }
    if (y > a.bottom - 4 && y <= ch.height && x >= a.left - 8 && x <= a.right + 8) {
      var xf = (x - a.left) / w;
      return (xf < 0.15 || xf > 0.85) ? 'x-bottom-edge' : 'x-bottom-mid';
    }
    if (x >= a.left && x <= a.right && y >= a.top && y <= a.bottom) {
      return 'plot';
    }
    return 'none';
  }

  function zoneFamily(z) {
    if (z === 'y-left-mid' || z === 'y-left-edge') return 'y-left';
    if (z === 'y-right-mid' || z === 'y-right-edge') return 'y-right';
    if (z === 'x-bottom-mid' || z === 'x-bottom-edge') return 'x-bottom';
    return z;
  }

  function makeOverlayPlugin(stateRef, opts) {
    return {
      id: 'axisZoomVisual_' + Math.random().toString(36).slice(2, 8),
      // Suppress noisy hover tooltips while dragging an axis or drawing a zoom box!
      beforeEvent: function (chart, args) {
        if (stateRef.drag && args.event && (args.event.type === 'mousemove' || args.event.type === 'mousedown')) {
          if (chart.tooltip && chart.tooltip.setActiveElements) {
            chart.tooltip.setActiveElements([], { x: 0, y: 0 });
          }
          return false;
        }
      },
      afterDraw: function (chart) {
        var a = chart.chartArea;
        if (!a) return;
        var ctx = chart.ctx;

        // 1. Draw active Box-Zoom selection rectangle
        if (stateRef.drag && stateRef.drag.mode === 'box') {
          var d = stateRef.drag;
          var x0 = Math.max(a.left, Math.min(d.x0, d.x1));
          var x1 = Math.min(a.right, Math.max(d.x0, d.x1));
          var y0 = opts.boxZoomMode === 'xy' ? Math.max(a.top, Math.min(d.y0, d.y1)) : a.top;
          var y1 = opts.boxZoomMode === 'xy' ? Math.min(a.bottom, Math.max(d.y0, d.y1)) : a.bottom;
          ctx.save();
          ctx.fillStyle = 'rgba(45, 114, 210, 0.16)';
          ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
          ctx.strokeStyle = '#2d72d2';
          ctx.lineWidth = 1.6;
          ctx.setLineDash([5, 3]);
          ctx.strokeRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
          ctx.setLineDash([]);
          ctx.restore();
          return;
        }

        // 2. Draw crisp hover rail + micro-hint badge on hovered axis so Shift vs Rescale is unmistakable!
        var z = stateRef.drag ? stateRef.drag.zone : stateRef.hoverZone;
        if (!z || z === 'none' || z === 'plot') return;
        var fam = zoneFamily(z);
        var isEdge = z.indexOf('-edge') > 0;

        ctx.save();
        ctx.fillStyle = isEdge ? 'rgba(217, 130, 43, 0.10)' : 'rgba(45, 114, 210, 0.09)';
        ctx.strokeStyle = isEdge ? 'rgba(217, 130, 43, 0.70)' : 'rgba(45, 114, 210, 0.65)';
        ctx.lineWidth = 2.2;

        var hintText = '';
        var hx = a.left + 8, hy = a.top + 8;

        if (fam === 'y-left') {
          ctx.fillRect(2, a.top, Math.max(0, a.left - 2), a.bottom - a.top);
          ctx.beginPath(); ctx.moveTo(a.left - 1, a.top); ctx.lineTo(a.left - 1, a.bottom); ctx.stroke();
          hintText = isEdge ? 'Left Y: Drag/Scroll = Rescale' : 'Left Y: Drag = Shift  |  Scroll = Rescale';
          hx = a.left + 8; hy = a.top + 8;
        } else if (fam === 'y-right') {
          ctx.fillRect(a.right, a.top, Math.max(0, chart.width - a.right - 2), a.bottom - a.top);
          ctx.beginPath(); ctx.moveTo(a.right + 1, a.top); ctx.lineTo(a.right + 1, a.bottom); ctx.stroke();
          hintText = isEdge ? 'Right Y: Drag/Scroll = Rescale' : 'Right Y: Drag = Shift  |  Scroll = Rescale';
          ctx.font = '600 10.5px -apple-system, BlinkMacSystemFont, sans-serif';
          hx = Math.max(a.left + 8, a.right - ctx.measureText(hintText).width - 20);
          hy = a.top + 8;
        } else if (fam === 'x-bottom') {
          ctx.fillRect(a.left, a.bottom, a.right - a.left, Math.max(0, chart.height - a.bottom - 2));
          ctx.beginPath(); ctx.moveTo(a.left, a.bottom + 1); ctx.lineTo(a.right, a.bottom + 1); ctx.stroke();
          hintText = isEdge ? 'X Axis: Drag/Scroll = Rescale' : 'X Axis: Drag = Shift  |  Scroll = Rescale';
          ctx.font = '600 10.5px -apple-system, BlinkMacSystemFont, sans-serif';
          hx = a.left + (a.right - a.left - ctx.measureText(hintText).width - 14) / 2;
          hy = Math.max(a.top + 6, a.bottom - 24);
        }

        if (hintText) {
          ctx.font = '600 10.5px -apple-system, BlinkMacSystemFont, sans-serif';
          var tw = ctx.measureText(hintText).width + 14;
          ctx.fillStyle = 'rgba(24, 32, 38, 0.86)';
          ctx.beginPath();
          ctx.roundRect ? ctx.roundRect(hx, hy, tw, 20, 4) : ctx.fillRect(hx, hy, tw, 20);
          ctx.fill();
          ctx.fillStyle = '#ffffff';
          ctx.fillText(hintText, hx + 7, hy + 13.5);
        }
        ctx.restore();
      }
    };
  }

  function attach(canvas, getChart, opts) {
    opts = opts || {};
    var xId = opts.xScaleId || 'x';
    var yLeftId = opts.yLeftScaleId || 'y';
    var yRightId = opts.rightScaleId || null;
    var wheelPlotMode = opts.wheelPlotMode || 'xy'; // 'x' or 'xy'
    var boxZoomMode = opts.boxZoomMode || 'x';      // 'x' or 'xy'

    var ui = { hoverZone: 'none', drag: null };
    if (canvas._axisZoomCleanup) canvas._axisZoomCleanup();

    function getPlotMode() {
      return opts.getPlotMode ? opts.getPlotMode() : 'box'; // 'box' | 'pan'
    }
    function getXLimits() { return opts.getXLimits ? opts.getXLimits() : null; }
    function getYLeftLimits() { return opts.getYLeftLimits ? opts.getYLeftLimits() : null; }
    function getYRightLimits() { return opts.getYRightLimits ? opts.getYRightLimits() : null; }

    function applyAxis(ch, scaleId, range) {
      if (!range || !isFinite(range[0]) || !isFinite(range[1]) || range[1] <= range[0]) return;
      var sc = ch.scales[scaleId];
      if (!sc) return;
      // Freeze any other currently-auto scales at their rendered bounds so Chart.js never auto-shrinks X when Y shifts!
      Object.keys(ch.scales || {}).forEach(function (otherId) {
        if (otherId === scaleId) return;
        var osc = ch.scales[otherId];
        var oopt = ch.options && ch.options.scales && ch.options.scales[otherId];
        if (osc && oopt && (oopt.min === undefined || oopt.max === undefined) && isFinite(osc.min) && isFinite(osc.max)) {
          oopt.min = osc.min;
          oopt.max = osc.max;
          osc.options.min = osc.min;
          osc.options.max = osc.max;
        }
      });
      sc.options.min = range[0];
      sc.options.max = range[1];
      if (sc.options.ticks) sc.options.ticks.includeBounds = false;
      if (ch.options && ch.options.scales && ch.options.scales[scaleId]) {
        ch.options.scales[scaleId].min = range[0];
        ch.options.scales[scaleId].max = range[1];
        ch.options.scales[scaleId].beginAtZero = false;
        if (!ch.options.scales[scaleId].ticks) ch.options.scales[scaleId].ticks = {};
        ch.options.scales[scaleId].ticks.includeBounds = false;
      }
      if (opts.onAxisChange) opts.onAxisChange(scaleId, range, false);
    }

    function cursorForZone(z) {
      if (z === 'y-left-mid' || z === 'y-right-mid' || z === 'x-bottom-mid') return 'grab';
      if (z === 'y-left-edge' || z === 'y-right-edge') return 'ns-resize';
      if (z === 'x-bottom-edge') return 'ew-resize';
      if (z === 'plot') {
        if (opts.getPlotCursor) return opts.getPlotCursor();
        return getPlotMode() === 'box' ? 'crosshair' : 'grab';
      }
      return 'default';
    }

    // Mouse Wheel: ALWAYS cleanly rescales the hovered axis (or plot)!
    function onWheel(e) {
      var ch = getChart();
      if (!ch || !ch.chartArea) return;
      var rect = canvas.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;
      var z = detectZone(ch, x, y, opts);
      if (z === 'none') return;

      e.preventDefault();
      e.stopPropagation();

      var fam = zoneFamily(z);
      var delta = Math.max(-180, Math.min(180, e.deltaY));
      var factor = Math.exp(delta * 0.0017);
      var a = ch.chartArea;

      if (fam === 'y-left') {
        var scL = ch.scales[yLeftId];
        if (!scL || !isFinite(scL.min) || !isFinite(scL.max)) return;
        var ancL = scL.getValueForPixel(Math.max(a.top, Math.min(a.bottom, y)));
        var rL = scaleRange(scL.min, scL.max, ancL, factor, isLog(scL), getYLeftLimits());
        if (rL) { applyAxis(ch, yLeftId, rL); ch.update('none'); }
        return;
      }
      if (fam === 'y-right' && yRightId && ch.scales[yRightId]) {
        var scR = ch.scales[yRightId];
        if (!isFinite(scR.min) || !isFinite(scR.max)) return;
        var ancR = scR.getValueForPixel(Math.max(a.top, Math.min(a.bottom, y)));
        var rR = scaleRange(scR.min, scR.max, ancR, factor, isLog(scR), getYRightLimits());
        if (rR) { applyAxis(ch, yRightId, rR); ch.update('none'); }
        return;
      }
      if (fam === 'x-bottom' || (fam === 'plot' && wheelPlotMode === 'x')) {
        var scX = ch.scales[xId];
        if (!scX || !isFinite(scX.min) || !isFinite(scX.max)) return;
        var ancX = scX.getValueForPixel(Math.max(a.left, Math.min(a.right, x)));
        var rX = scaleRange(scX.min, scX.max, ancX, factor, isLog(scX), getXLimits());
        if (rX) {
          applyAxis(ch, xId, rX);
          ch.update('none');
          if (opts.onXViewCommit) opts.onXViewCommit(rX);
        }
        return;
      }
      if (fam === 'plot' && wheelPlotMode === 'xy') {
        var sX = ch.scales[xId], sY = ch.scales[yLeftId];
        var changed = false;
        if (sX && isFinite(sX.min) && isFinite(sX.max)) {
          var nX = scaleRange(sX.min, sX.max, sX.getValueForPixel(x), factor, isLog(sX), getXLimits());
          if (nX) { applyAxis(ch, xId, nX); changed = true; }
        }
        if (sY && isFinite(sY.min) && isFinite(sY.max)) {
          var nY = scaleRange(sY.min, sY.max, sY.getValueForPixel(y), factor, isLog(sY), getYLeftLimits());
          if (nY) { applyAxis(ch, yLeftId, nY); changed = true; }
        }
        if (changed) ch.update('none');
      }
    }

    function onMouseDown(e) {
      if (e.button !== 0) return;
      var ch = getChart();
      if (!ch || !ch.chartArea) return;
      var rect = canvas.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;
      var z = detectZone(ch, x, y, opts);
      if (z === 'none') return;

      if (z === 'plot' && opts.shouldIgnorePlotDrag && opts.shouldIgnorePlotDrag(e, ch, x, y)) {
        return;
      }

      e.preventDefault();
      var a = ch.chartArea;
      var fam = zoneFamily(z);

      // 1. Axis SHIFT (middle drag) vs Axis RESCALE (edge drag)
      if (fam === 'y-left' || fam === 'y-right') {
        var sid = fam === 'y-left' ? yLeftId : yRightId;
        var sc = ch.scales[sid];
        if (!sc || !isFinite(sc.min) || !isFinite(sc.max)) return;
        ui.drag = {
          mode: z.indexOf('-edge') > 0 ? 'y-rescale' : 'y-shift',
          zone: z, scaleId: sid, x0: x, y0: y,
          min0: sc.min, max0: sc.max, log: isLog(sc),
          anchor: sc.getValueForPixel((a.top + a.bottom) / 2),
          limits: fam === 'y-left' ? getYLeftLimits() : getYRightLimits(),
          topEdge: (y - a.top) < (a.bottom - a.top) / 2
        };
        canvas.style.cursor = z.indexOf('-edge') > 0 ? 'ns-resize' : 'grabbing';
        return;
      }

      if (fam === 'x-bottom') {
        var scX = ch.scales[xId];
        if (!scX || !isFinite(scX.min) || !isFinite(scX.max)) return;
        ui.drag = {
          mode: z.indexOf('-edge') > 0 ? 'x-rescale' : 'x-shift',
          zone: z, scaleId: xId, x0: x, y0: y,
          min0: scX.min, max0: scX.max, log: isLog(scX),
          rev: Boolean(scX.options && scX.options.reverse),
          anchor: scX.getValueForPixel((a.left + a.right) / 2),
          limits: getXLimits()
        };
        canvas.style.cursor = z.indexOf('-edge') > 0 ? 'ew-resize' : 'grabbing';
        return;
      }

      // 2. Plot Area: Box Zoom OR Pan (Shift+Drag always forces Box Zoom even in Pan mode!)
      if (z === 'plot') {
        var useBox = e.shiftKey || getPlotMode() === 'box';
        if (useBox) {
          ui.drag = { mode: 'box', zone: 'plot', x0: x, y0: y, x1: x, y1: y };
          canvas.style.cursor = 'crosshair';
        } else {
          var sx = ch.scales[xId], sy = ch.scales[yLeftId];
          if (!sx || !isFinite(sx.min) || !isFinite(sx.max)) return;
          ui.drag = {
            mode: 'pan', zone: 'plot', x0: x, y0: y,
            xMin0: sx.min, xMax0: sx.max, xLog: isLog(sx), xRev: Boolean(sx.options && sx.options.reverse), xLimits: getXLimits(),
            yMin0: sy ? sy.min : 0, yMax0: sy ? sy.max : 1, yLog: sy ? isLog(sy) : false, yLimits: getYLeftLimits()
          };
          canvas.style.cursor = 'grabbing';
        }
      }
    }

    function onWindowMouseMove(e) {
      var ch = getChart();
      if (!ch || !ch.chartArea) return;
      var rect = canvas.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;

      if (!ui.drag) {
        var z = detectZone(ch, x, y, opts);
        if (z !== ui.hoverZone) {
          ui.hoverZone = z;
          canvas.style.cursor = cursorForZone(z);
          ch.draw();
        }
        return;
      }

      var d = ui.drag;
      var a = ch.chartArea;
      var plotH = Math.max(1, a.bottom - a.top);
      var plotW = Math.max(1, a.right - a.left);

      if (d.mode === 'box') {
        d.x1 = x; d.y1 = y;
        ch.draw();
        return;
      }

      if (d.mode === 'y-shift') {
        // Drag UP (y < y0) -> plot data moves UP -> axis min & max DECREASE!
        var fracY = (d.y0 - y) / plotH;
        var rYS = shiftLinearOrLog(d.min0, d.max0, fracY, d.log, null);
        if (rYS) { applyAxis(ch, d.scaleId, rYS); ch.update('none'); }
        return;
      }

      if (d.mode === 'y-rescale') {
        var dy = y - d.y0;
        var sign = d.topEdge ? 1 : -1;
        var fY = Math.exp(sign * dy * 0.0085);
        var rYR = scaleRange(d.min0, d.max0, d.anchor, fY, d.log, d.limits);
        if (rYR) { applyAxis(ch, d.scaleId, rYR); ch.update('none'); }
        return;
      }

      if (d.mode === 'x-shift') {
        var fracX = (x - d.x0) / plotW;
        if (d.rev) fracX = -fracX;
        var rXS = shiftLinearOrLog(d.min0, d.max0, fracX, d.log, d.limits);
        if (rXS) { applyAxis(ch, d.scaleId, rXS); d.lastXRange = rXS; ch.update('none'); }
        return;
      }

      if (d.mode === 'x-rescale') {
        var dx = x - d.x0;
        var fX = Math.exp(-dx * 0.0065);
        var rXR = scaleRange(d.min0, d.max0, d.anchor, fX, d.log, d.limits);
        if (rXR) { applyAxis(ch, d.scaleId, rXR); d.lastXRange = rXR; ch.update('none'); }
        return;
      }

      if (d.mode === 'pan') {
        var fx = (x - d.x0) / plotW;
        if (d.xRev) fx = -fx;
        var nx = shiftLinearOrLog(d.xMin0, d.xMax0, fx, d.xLog, d.xLimits);
        if (nx) { applyAxis(ch, xId, nx); d.lastXRange = nx; }

        // Always allow vertical shift of Left Y when panning the plot!
        var fy = (d.y0 - y) / plotH;
        var ny = shiftLinearOrLog(d.yMin0, d.yMax0, fy, d.yLog, null);
        if (ny) applyAxis(ch, yLeftId, ny);

        ch.update('none');
      }
    }

    function onWindowMouseUp() {
      if (!ui.drag) return;
      var d = ui.drag;
      ui.drag = null;
      var ch = getChart();

      if (d.mode === 'box' && ch && ch.chartArea) {
        var a = ch.chartArea;
        var px0 = Math.max(a.left, Math.min(d.x0, d.x1));
        var px1 = Math.min(a.right, Math.max(d.x0, d.x1));
        var py0 = Math.max(a.top, Math.min(d.y0, d.y1));
        var py1 = Math.min(a.bottom, Math.max(d.y0, d.y1));
        if ((px1 - px0) > 10 && (boxZoomMode === 'x' || (py1 - py0) > 10)) {
          var sx = ch.scales[xId], sy = ch.scales[yLeftId];
          if (sx) {
            var vx0 = sx.getValueForPixel(px0), vx1 = sx.getValueForPixel(px1);
            var rx = [Math.min(vx0, vx1), Math.max(vx0, vx1)];
            var xLim = getXLimits();
            if (!xLim || !xLim.minSpan || (rx[1] - rx[0]) >= xLim.minSpan) {
              applyAxis(ch, xId, rx);
              if (opts.onXViewCommit) opts.onXViewCommit(rx);
            }
          }
          if (boxZoomMode === 'xy' && sy && (py1 - py0) > 10) {
            var vy0 = sy.getValueForPixel(py0), vy1 = sy.getValueForPixel(py1);
            var ry = [Math.min(vy0, vy1), Math.max(vy0, vy1)];
            applyAxis(ch, yLeftId, ry);
          }
          ch.update('none');
        } else {
          ch.draw();
        }
      } else if (d.lastXRange && opts.onXViewCommit) {
        opts.onXViewCommit(d.lastXRange);
      }

      canvas.style.cursor = cursorForZone(ui.hoverZone);
    }

    function onMouseLeave() {
      if (!ui.drag && ui.hoverZone !== 'none') {
        ui.hoverZone = 'none';
        var ch = getChart();
        if (ch) ch.draw();
      }
    }

    function onDblClick(e) {
      var ch = getChart();
      if (!ch || !ch.chartArea) return;
      var rect = canvas.getBoundingClientRect();
      var z = detectZone(ch, e.clientX - rect.left, e.clientY - rect.top, opts);
      var fam = zoneFamily(z);
      if (opts.onReset) {
        opts.onReset(fam);
        return;
      }
      if (fam === 'y-left' && ch.options.scales[yLeftId]) {
        delete ch.options.scales[yLeftId].min; delete ch.options.scales[yLeftId].max;
        if (ch.options.scales[yLeftId].ticks) ch.options.scales[yLeftId].ticks.includeBounds = true;
        ch.update('none');
      } else if (fam === 'y-right' && yRightId && ch.options.scales[yRightId]) {
        delete ch.options.scales[yRightId].min; delete ch.options.scales[yRightId].max;
        if (ch.options.scales[yRightId].ticks) ch.options.scales[yRightId].ticks.includeBounds = true;
        ch.update('none');
      } else if (fam === 'x-bottom' && ch.options.scales[xId]) {
        delete ch.options.scales[xId].min; delete ch.options.scales[xId].max;
        ch.update('none');
      } else {
        Object.keys(ch.options.scales || {}).forEach(function (k) {
          delete ch.options.scales[k].min; delete ch.options.scales[k].max;
          if (ch.options.scales[k].ticks) ch.options.scales[k].ticks.includeBounds = true;
        });
        ch.update('none');
      }
    }

    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mouseleave', onMouseLeave);
    canvas.addEventListener('dblclick', onDblClick);
    window.addEventListener('mousemove', onWindowMouseMove);
    window.addEventListener('mouseup', onWindowMouseUp);

    var hoverPlugin = makeOverlayPlugin(ui, { boxZoomMode: boxZoomMode });
    canvas._axisZoomCleanup = function () {
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('mousedown', onMouseDown);
      canvas.removeEventListener('mouseleave', onMouseLeave);
      canvas.removeEventListener('dblclick', onDblClick);
      window.removeEventListener('mousemove', onWindowMouseMove);
      window.removeEventListener('mouseup', onWindowMouseUp);
    };

    return { hoverPlugin: hoverPlugin, ui: ui };
  }

  window.AxisZoom = { attach: attach };
})();
