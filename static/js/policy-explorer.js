/* Interactive replay of the Van der Pol policy training run.
   Replay mode scrubs the recorded epochs; explore mode hands the policy
   coefficients, the plant and the initial conditions over to the reader. */
(function () {
  "use strict";

  var root = document.getElementById("policy-explorer");
  if (!root) return;

  var DT = 0.02, T = 10, STEPS = Math.round(T / DT), BOX = 4.4;
  var D = null, frame = 0, playing = false, last = 0, raf = 0, explore = false;
  var S = { mu: 1, umax: 5, nic: 8, spread: 4.2, coef: null };
  var view = { azim: -56, elev: 26 };

  var el = function (id) { return document.getElementById(id); };
  var slider = el("pe-epoch"), play = el("pe-play"), readout = el("pe-readout");
  var mode = el("pe-mode"), reset = el("pe-reset");
  var cStates = el("pe-states"), cMap = el("pe-map"), cLoss = el("pe-loss");
  var coefCells = [];
  var css = function (n) { return getComputedStyle(document.body).getPropertyValue(n).trim(); };
  var PLANT = [["mu", 2], ["umax", 1], ["nic", 0], ["spread", 1]];
  var TERMS_TEX = ["x_1", "x_2", "x_1^2", "x_1x_2", "x_2^2",
    "x_1^3", "x_1^2x_2", "x_1x_2^2", "x_2^3",
    "x_1^4", "x_1^3x_2", "x_1^2x_2^2", "x_1x_2^3", "x_2^4"];
  var termNodes = [];
  function typeset(nodes) {
    if (!window.MathJax) return;
    var go = function () { if (MathJax.typesetPromise) MathJax.typesetPromise(nodes); };
    (MathJax.startup && MathJax.startup.promise) ? MathJax.startup.promise.then(go) : go();
  }
  function buildTermLabels() {
    var box = el("pe-coef-terms");
    if (!box) return;
    box.innerHTML = "";
    termNodes = TERMS_TEX.map(function (t) {
      var span = document.createElement("span");
      span.innerHTML = "\\(" + t + "\\)";
      box.appendChild(span);
      return span;
    });
    typeset([box]);
  }

  /* Canvas cannot host MathJax, so axis labels are drawn in a math serif
     with real subscripts, matching the LaTeX look of the typeset text. */
  var MATH_STACK = '"STIX Two Math","Latin Modern Math","Cambria Math","Times New Roman",Times,serif';
  function mathFont(size, italic, weight) {
    return (italic ? "italic " : "") + (weight || 400) + " " + size + "px " + MATH_STACK;
  }
  function mathLabel(ctx, parts, x, y, size, align) {
    var w = 0, i, p;
    for (i = 0; i < parts.length; i++) {
      p = parts[i];
      ctx.font = mathFont(p.sub ? size * 0.72 : size, p.up ? false : true, p.w);
      w += ctx.measureText(p.t).width;
    }
    var cx = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
    for (i = 0; i < parts.length; i++) {
      p = parts[i];
      ctx.font = mathFont(p.sub ? size * 0.72 : size, p.up ? false : true, p.w);
      ctx.textAlign = "left";
      ctx.fillText(p.t, cx, y + (p.sub ? size * 0.24 : 0));
      cx += ctx.measureText(p.t).width;
    }
  }
  var SYM = {
    x1: [{ t: "x", w: 500 }, { t: "1", sub: true, up: true }],
    x2: [{ t: "x", w: 500 }, { t: "2", sub: true, up: true }],
    x: [{ t: "x", w: 500 }],
    u: [{ t: "u", w: 500 }],
    t: [{ t: "t", w: 500 }, { t: " [s]", up: true, w: 400 }]
  };

  function features(x1, x2) {
    return [x1, x2, x1 * x1, x1 * x2, x2 * x2,
            x1 * x1 * x1, x1 * x1 * x2, x1 * x2 * x2, x2 * x2 * x2,
            Math.pow(x1, 4), x1 * x1 * x1 * x2, x1 * x1 * x2 * x2,
            x1 * x2 * x2 * x2, Math.pow(x2, 4)];
  }
  function policy(c, x1, x2) {
    var f = features(x1, x2), u = 0;
    for (var i = 0; i < 14; i++) u += c[i] * f[i];
    return u > S.umax ? S.umax : (u < -S.umax ? -S.umax : u);
  }
  function deriv(x1, x2, u) { return [x2, S.mu * (1 - x1 * x1) * x2 - x1 + u]; }

  function rollout(c, x0) {
    var xs = new Float64Array(2 * (STEPS + 1)), us = new Float64Array(STEPS);
    var x1 = x0[0], x2 = x0[1], k1, k2, k3, k4, u, i;
    xs[0] = x1; xs[1] = x2;
    for (i = 0; i < STEPS; i++) {
      u = policy(c, x1, x2); us[i] = u;
      k1 = deriv(x1, x2, u);
      k2 = deriv(x1 + 0.5 * DT * k1[0], x2 + 0.5 * DT * k1[1], u);
      k3 = deriv(x1 + 0.5 * DT * k2[0], x2 + 0.5 * DT * k2[1], u);
      k4 = deriv(x1 + DT * k3[0], x2 + DT * k3[1], u);
      x1 += DT * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
      x2 += DT * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
      if (!isFinite(x1) || Math.abs(x1) > 1e4) { x1 = NaN; x2 = NaN; }
      xs[2 * (i + 1)] = x1; xs[2 * (i + 1) + 1] = x2;
    }
    return { xs: xs, us: us };
  }

  function initials() {
    if (S.nic === 1) return [[-S.spread * 0.7, S.spread * 0.7]];
    var out = [], i;
    for (i = 0; i < S.nic; i++) {
      var th = 2 * Math.PI * i / S.nic + Math.PI / S.nic;
      out.push([S.spread * Math.cos(th), S.spread * Math.sin(th)]);
    }
    return out;
  }
  function coefNow() { return S.coef || D.coef[frame]; }

  function fit(canvas, h) {
    var r = window.devicePixelRatio || 1, w = canvas.parentNode.clientWidth;
    canvas.width = Math.round(w * r); canvas.height = Math.round(h * r);
    canvas.style.width = "100%"; canvas.style.height = h + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }
  var RAMP = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  function ramp(t) {
    t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    var i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i, A = RAMP[i], B = RAMP[i + 1];
    return "rgb(" + Math.round(A[0] + f * (B[0] - A[0])) + "," +
                    Math.round(A[1] + f * (B[1] - A[1])) + "," +
                    Math.round(A[2] + f * (B[2] - A[2])) + ")";
  }
  var COLORS = ["#463d86", "#168a79", "#c1612a", "#9a3b74",
                "#2f6fae", "#6a8e2b", "#b03030", "#4f7f8f"];
  function nice(v) { return Math.abs(v % 1) < 1e-9 ? String(v) : v.toFixed(1); }

  function drawStates(trajs) {
    var g = fit(cStates, 282), ctx = g.ctx, P = 42, R = 10, gap = 22;
    var hh = (g.h - 30 - gap) / 2, y1 = 6, y2 = 6 + hh + gap;
    var xs = function (t) { return P + t / T * (g.w - P - R); };
    var ys = function (v, y0, s) { return y0 + hh / 2 - (v / s) * (hh / 2); };
    var SX = BOX, SU = S.umax * 1.25;

    [[y1, SX, [-4, 0, 4], "x"], [y2, SU, [-S.umax, 0, S.umax], "u"]].forEach(function (p) {
      ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
      ctx.strokeRect(P, p[0], g.w - P - R, hh);
      ctx.font = "11px " + css("--sans"); ctx.textAlign = "right";
      p[2].forEach(function (v) {
        var Y = ys(v, p[0], p[1]);
        ctx.fillStyle = css("--ink-3"); ctx.fillText(nice(v), P - 6, Y + 4);
        ctx.strokeStyle = css("--line-soft");
        ctx.beginPath(); ctx.moveTo(P, Y); ctx.lineTo(P + 4, Y); ctx.stroke();
      });
      ctx.fillStyle = css("--ink-2");
      ctx.save(); ctx.translate(12, p[0] + hh / 2); ctx.rotate(-Math.PI / 2);
      mathLabel(ctx, SYM[p[3]], 0, 4, 15, "center"); ctx.restore();
    });

    ctx.strokeStyle = css("--danger"); ctx.setLineDash([4, 4]); ctx.globalAlpha = 0.75;
    [1, -1].forEach(function (s) {
      var Y = ys(s * S.umax, y2, SU);
      ctx.beginPath(); ctx.moveTo(P, Y); ctx.lineTo(g.w - R, Y); ctx.stroke();
    });
    ctx.globalAlpha = 0.3; ctx.strokeStyle = css("--ink-3");
    ctx.beginPath(); ctx.moveTo(P, ys(0, y1, SX)); ctx.lineTo(g.w - R, ys(0, y1, SX)); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    trajs.forEach(function (tr, i) {
      ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 1.4;
      [[0, []], [1, [5, 3]]].forEach(function (c) {
        ctx.setLineDash(c[1]); ctx.beginPath();
        for (var s = 0; s <= STEPS; s += 4) {
          var v = tr.xs[2 * s + c[0]]; if (!isFinite(v)) break;
          var Y = Math.max(y1 - 3, Math.min(y1 + hh + 3, ys(v, y1, SX)));
          s ? ctx.lineTo(xs(s * DT), Y) : ctx.moveTo(xs(s * DT), Y);
        }
        ctx.stroke();
      });
      ctx.setLineDash([]); ctx.beginPath();
      for (var s2 = 0; s2 < STEPS; s2 += 4) {
        var Y2 = Math.max(y2 - 3, Math.min(y2 + hh + 3, ys(tr.us[s2], y2, SU)));
        s2 ? ctx.lineTo(xs(s2 * DT), Y2) : ctx.moveTo(xs(s2 * DT), Y2);
      }
      ctx.stroke();
    });

    ctx.fillStyle = css("--ink-3"); ctx.font = "11px " + css("--sans"); ctx.textAlign = "center";
    for (var t = 0; t <= T; t += 2) ctx.fillText(t, xs(t), g.h - 10);
    ctx.fillStyle = css("--ink-2");
    mathLabel(ctx, SYM.t, (P + g.w - R) / 2, g.h, 14, "center");
  }

  function project(X, Y, Z, P) {
    var az = view.azim * Math.PI / 180, ev = view.elev * Math.PI / 180;
    var sx = -Math.sin(az) * X + Math.cos(az) * Y;
    var sy = -Math.cos(az) * Math.sin(ev) * X - Math.sin(az) * Math.sin(ev) * Y + Math.cos(ev) * Z;
    var depth = Math.cos(ev) * Math.cos(az) * X + Math.cos(ev) * Math.sin(az) * Y + Math.sin(ev) * Z;
    return [P.cx + sx * P.s, P.cy - sy * P.s, depth];
  }

  function drawSurface(c) {
    var g = fit(cMap, 282), ctx = g.ctx, BARW = 40;
    var P = { cx: (g.w - BARW) / 2, cy: g.h / 2 + 2, s: Math.min(g.w - BARW, g.h) * 0.36 };
    var N = 26, Zs = [], i, j, ZB = 0.55;
    for (j = 0; j < N; j++) {
      Zs.push([]);
      for (i = 0; i < N; i++) {
        Zs[j].push(policy(c, -BOX + 2 * BOX * i / (N - 1), -BOX + 2 * BOX * j / (N - 1)));
      }
    }
    var nx = function (q) { return -1 + 2 * q / (N - 1); };
    var nz = function (u) { return ZB * u / S.umax; };

    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    [[-1, -1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [-1, 1, -1, -1]].forEach(function (e) {
      var A = project(e[0], e[1], -ZB, P), B = project(e[2], e[3], -ZB, P);
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
    });
    var zTop = project(-1, -1, ZB, P), zBot = project(-1, -1, -ZB, P);
    ctx.beginPath(); ctx.moveTo(zBot[0], zBot[1]); ctx.lineTo(zTop[0], zTop[1]); ctx.stroke();

    var quads = [];
    for (j = 0; j < N - 1; j++) for (i = 0; i < N - 1; i++) {
      var p00 = project(nx(i), nx(j), nz(Zs[j][i]), P);
      var p10 = project(nx(i + 1), nx(j), nz(Zs[j][i + 1]), P);
      var p11 = project(nx(i + 1), nx(j + 1), nz(Zs[j + 1][i + 1]), P);
      var p01 = project(nx(i), nx(j + 1), nz(Zs[j + 1][i]), P);
      quads.push({
        p: [p00, p10, p11, p01],
        d: (p00[2] + p10[2] + p11[2] + p01[2]) / 4,
        u: (Zs[j][i] + Zs[j][i + 1] + Zs[j + 1][i + 1] + Zs[j + 1][i]) / 4
      });
    }
    quads.sort(function (A, B) { return A.d - B.d; });
    quads.forEach(function (q) {
      ctx.fillStyle = ramp((q.u + S.umax) / (2 * S.umax));
      ctx.strokeStyle = ctx.fillStyle; ctx.lineWidth = 0.6;
      ctx.beginPath(); ctx.moveTo(q.p[0][0], q.p[0][1]);
      ctx.lineTo(q.p[1][0], q.p[1][1]); ctx.lineTo(q.p[2][0], q.p[2][1]);
      ctx.lineTo(q.p[3][0], q.p[3][1]); ctx.closePath(); ctx.fill(); ctx.stroke();
    });

    ctx.fillStyle = css("--ink-3"); ctx.font = "10.5px " + css("--sans"); ctx.textAlign = "center";
    [-4, 0, 4].forEach(function (v) {
      var t = v / BOX;
      var a = project(t, -1.18, -ZB, P), b = project(1.18, t, -ZB, P);
      ctx.fillText(v, a[0], a[1] + 4);
      ctx.fillText(v, b[0], b[1] + 4);
    });
    ctx.textAlign = "right";
    [-S.umax, 0, S.umax].forEach(function (v) {
      var z = project(-1.1, -1.1, nz(v), P);
      ctx.fillText(nice(v), z[0], z[1] + 4);
    });
    ctx.fillStyle = css("--ink-2");
    var xl = project(0, -1.5, -ZB, P), yl = project(1.5, 0, -ZB, P);
    mathLabel(ctx, SYM.x1, xl[0], xl[1] + 5, 15, "center");
    mathLabel(ctx, SYM.x2, yl[0], yl[1] + 5, 15, "center");

    var bx = g.w - BARW + 10, by = 18, bh = g.h - 48;
    for (var y = 0; y < bh; y++) { ctx.fillStyle = ramp(1 - y / bh); ctx.fillRect(bx, by + y, 12, 1); }
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, 12, bh);
    ctx.fillStyle = css("--ink-3"); ctx.font = "10.5px " + css("--sans"); ctx.textAlign = "right";
    ctx.fillText(nice(S.umax), bx + 12, by - 5);
    ctx.fillText(nice(-S.umax), bx + 12, by + bh + 12);
    ctx.save(); ctx.translate(bx + 28, by + bh / 2); ctx.rotate(Math.PI / 2);
    ctx.fillStyle = css("--ink-2");
    mathLabel(ctx, SYM.u, 0, 4, 15, "center"); ctx.restore();
  }

  function buildCoefRow() {
    var box = el("pe-coef-row");
    if (!box) return;
    box.innerHTML = "";
    coefCells = [];
    for (var i = 0; i < 14; i++) {
      var cell = document.createElement("div");
      cell.className = "pe-coef-cellbox";
      box.appendChild(cell);
      coefCells.push(cell);
    }
  }

  function drawCoef(c, mask) {
    var lim = 3;
    for (var i = 0; i < 14; i++) {
      var cell = coefCells[i];
      if (!cell) continue;
      var v = c[i], zero = Math.abs(v) < 1e-9;
      var t = Math.max(-1, Math.min(1, v / lim)), r, gr, b;
      if (t >= 0) { r = 255; gr = Math.round(255 - 150 * t); b = Math.round(255 - 180 * t); }
      else { r = Math.round(255 + 190 * t); gr = Math.round(255 + 130 * t); b = 255; }
      cell.style.background = zero ? "" : "rgb(" + r + "," + gr + "," + b + ")";
      cell.className = "pe-coef-cellbox" + (zero ? " is-zero" : "") +
        (Math.abs(t) > 0.55 ? " is-strong" : "");
      cell.textContent = Math.abs(v) < 0.005 ? "0" : v.toFixed(2);
      if (termNodes[i]) termNodes[i].className = mask[i] ? "is-on" : "is-off";
    }
  }

  /* Proposition 3: the sensitivity certificate is a function of the retained
     coefficients alone. L_i is the largest gradient norm of the i-th library
     term on the box, so L_pi = sum_i |xi_i| L_i for this single-input policy. */
  var POWERS = [[1,0],[0,1],[2,0],[1,1],[0,2],[3,0],[2,1],[1,2],[0,3],[4,0],[3,1],[2,2],[1,3],[0,4]];
  var TERM_L = POWERS.map(function (p) {
    return Math.pow(BOX, p[0] + p[1] - 1) * Math.sqrt(p[0] * p[0] + p[1] * p[1]);
  });
  function lipCertified(c) {
    var s = 0;
    for (var i = 0; i < 14; i++) s += Math.abs(c[i]) * TERM_L[i];
    return s;
  }
  function gradNorm(c, x1, x2) {
    var g1 = 0, g2 = 0;
    for (var i = 0; i < 14; i++) {
      if (!c[i]) continue;
      var a = POWERS[i][0], b = POWERS[i][1];
      if (a > 0) g1 += c[i] * a * Math.pow(x1, a - 1) * Math.pow(x2, b);
      if (b > 0) g2 += c[i] * b * Math.pow(x1, a) * Math.pow(x2, b - 1);
    }
    return Math.sqrt(g1 * g1 + g2 * g2);
  }
  function lipMeasured(c, n) {
    n = n || 33;
    var m = 0;
    for (var i = 0; i < n; i++) {
      var x1 = -BOX + 2 * BOX * i / (n - 1);
      for (var j = 0; j < n; j++) {
        var g = gradNorm(c, x1, -BOX + 2 * BOX * j / (n - 1));
        if (g > m) m = g;
      }
    }
    return m;
  }
  var lipTrace = null;
  function buildLipTrace() {
    lipTrace = D.coef.map(function (c) { return [lipCertified(c), lipMeasured(c, 21)]; });
  }

  function drawLip(c, epochNow, live) {
    var cv = el("pe-lip");
    if (!cv || !lipTrace) return;
    var g = fit(cv, 120), ctx = g.ctx, P = 46, R = 92, top = 10, h = g.h - 28;
    var cert = lipCertified(c), meas = lipMeasured(c, 33);
    var hi = 0;
    lipTrace.forEach(function (p) { if (p[0] > hi) hi = p[0]; });
    hi = Math.max(hi, cert) * 1.05;
    var e0 = D.epoch[0], e1 = D.epoch[D.epoch.length - 1];
    var sx = function (e) { return P + (e - e0) / (e1 - e0) * (g.w - P - R); };
    var sy = function (v) { return top + h - Math.min(1, v / hi) * h; };

    ctx.fillStyle = css("--surface-2");
    ctx.fillRect(sx(D.debias), top, g.w - R - sx(D.debias), h);
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.strokeRect(P, top, g.w - P - R, h);
    D.prune.forEach(function (pe, i) {
      ctx.strokeStyle = css("--danger"); ctx.globalAlpha = 0.8; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx(pe), top); ctx.lineTo(sx(pe), top + h); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = css("--danger"); ctx.font = "600 10px " + css("--sans"); ctx.textAlign = "left";
      ctx.fillText("P" + (i + 1), sx(pe) + 3, top + 11);
    });
    [[0, css("--brand"), []], [1, css("--brand-2"), [5, 3]]].forEach(function (s) {
      ctx.strokeStyle = s[1]; ctx.lineWidth = 1.8; ctx.setLineDash(s[2]); ctx.beginPath();
      lipTrace.forEach(function (p, i) {
        var X = sx(D.epoch[i]), Y = sy(p[s[0]]);
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      });
      ctx.stroke(); ctx.setLineDash([]);
    });
    if (!live) {
      ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(sx(epochNow), top); ctx.lineTo(sx(epochNow), top + h); ctx.stroke();
    }
    ctx.fillStyle = css("--ink-3"); ctx.font = "11px " + css("--sans"); ctx.textAlign = "right";
    [0, Math.round(hi / 2), Math.round(hi)].forEach(function (v) { ctx.fillText(v, P - 5, sy(v) + 4); });
    ctx.textAlign = "center";
    [0, 2000, 4000].forEach(function (e) { ctx.fillText(e, sx(e), g.h - 2); });

    var bx = g.w - R + 12, by = top + 6;
    ctx.textAlign = "left";
    ctx.fillStyle = css("--brand"); ctx.font = "600 15px " + css("--math");
    ctx.fillText(cert.toFixed(1), bx, by + 14);
    ctx.fillStyle = css("--ink-3"); ctx.font = "10px " + css("--sans");
    ctx.fillText("certified", bx, by + 27);
    ctx.fillStyle = css("--brand-2"); ctx.font = "600 15px " + css("--math");
    ctx.fillText(meas.toFixed(1), bx, by + 50);
    ctx.fillStyle = css("--ink-3"); ctx.font = "10px " + css("--sans");
    ctx.fillText("measured", bx, by + 63);
    ctx.fillStyle = css("--ink-2"); ctx.font = "600 11px " + css("--sans");
    ctx.fillText((cert / Math.max(meas, 1e-6)).toFixed(2) + "\u00d7 slack", bx, by + 86);
  }

  function drawLoss(epochNow) {
    var g = fit(cLoss, 120), ctx = g.ctx, P = 42, R = 10, top = 8, h = g.h - 26;
    var e0 = D.lossEpoch[0], e1 = D.lossEpoch[D.lossEpoch.length - 1], lo = 20, hi = 130;
    var sx = function (e) { return P + (e - e0) / (e1 - e0) * (g.w - P - R); };
    var sy = function (v) {
      return top + h - (Math.log(Math.max(v, lo)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * h;
    };
    ctx.fillStyle = css("--surface-2");
    ctx.fillRect(sx(D.debias), top, g.w - R - sx(D.debias), h);
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.strokeRect(P, top, g.w - P - R, h);
    D.prune.forEach(function (pe, i) {
      ctx.strokeStyle = css("--danger"); ctx.globalAlpha = 0.85; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx(pe), top); ctx.lineTo(sx(pe), top + h); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = css("--danger"); ctx.font = "600 10px " + css("--sans"); ctx.textAlign = "left";
      ctx.fillText("P" + (i + 1), sx(pe) + 3, top + 11);
    });
    [["lossTrain", css("--brand"), []], ["lossVal", css("--brand-2"), [5, 3]]].forEach(function (s) {
      ctx.strokeStyle = s[1]; ctx.lineWidth = 1.6; ctx.setLineDash(s[2]); ctx.beginPath();
      D[s[0]].forEach(function (v, i) {
        var X = sx(D.lossEpoch[i]), Y = sy(v);
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      });
      ctx.stroke(); ctx.setLineDash([]);
    });
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(sx(epochNow), top); ctx.lineTo(sx(epochNow), top + h); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.font = "11px " + css("--sans"); ctx.textAlign = "center";
    [0, 1000, 2000, 3000, 4000].forEach(function (e) { ctx.fillText(e, sx(e), g.h - 2); });
    ctx.textAlign = "right";
    [20, 50, 100].forEach(function (v) { ctx.fillText(v, P - 5, sy(v) + 4); });
  }

  var EDITED_MASK = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
  function render() {
    var c = coefNow();
    drawStates(initials().map(function (x0) { return rollout(c, x0); }));
    drawSurface(c);
    drawCoef(c, explore ? EDITED_MASK : D.mask[frame]);
    if (!explore) drawLoss(D.epoch[frame]);
    drawLip(c, explore ? D.epoch[frame] : D.epoch[frame], explore);
    if (explore) {
      readout.innerHTML = "your policy &middot; μ = <strong>" + S.mu.toFixed(2) +
        "</strong> &middot; u<sub>max</sub> = <strong>" + S.umax.toFixed(1) +
        "</strong> &middot; <strong>" + S.nic + "</strong> initial condition" + (S.nic === 1 ? "" : "s");
    } else {
      readout.innerHTML = "epoch <strong>" + D.epoch[frame] + "</strong> &middot; <strong>" +
        D.active[frame] + "/14</strong> terms retained &middot; " +
        (D.epoch[frame] >= D.debias ? "debias" : "sparse") + " phase";
    }
  }

  function step(ts) {
    if (!playing) return;
    if (ts - last > 55) { last = ts; frame = (frame + 1) % D.frames; slider.value = frame; render(); }
    raf = requestAnimationFrame(step);
  }
  function setPlaying(on) {
    playing = on;
    play.textContent = on ? "Pause" : "Play";
    play.setAttribute("aria-pressed", on ? "true" : "false");
    if (on) { last = 0; raf = requestAnimationFrame(step); } else cancelAnimationFrame(raf);
  }

  var editInputs = [];
  function buildEditor() {
    var wrap = el("pe-edit");
    wrap.innerHTML = "";
    D.termsPlain.forEach(function (name, i) {
      var cell = document.createElement("label");
      cell.className = "pe-edit-cell";
      var span = document.createElement("span");
      span.innerHTML = "\\(" + TERMS_TEX[i] + "\\)";
      var input = document.createElement("input");
      input.type = "number"; input.step = "0.01"; input.id = "pe-c" + i;
      input.setAttribute("aria-label", "coefficient of " + name);
      input.addEventListener("input", function () {
        takeOver();
        var v = parseFloat(input.value);
        S.coef[i] = isFinite(v) ? v : 0;
        cell.classList.toggle("is-set", Math.abs(S.coef[i]) > 1e-9);
        render();
      });
      cell.appendChild(input); cell.appendChild(span);
      wrap.appendChild(cell);
      editInputs.push({ input: input, cell: cell });
    });
    typeset([wrap]);
  }
  function syncKnobs() {
    var c = coefNow();
    editInputs.forEach(function (e, i) {
      e.input.value = Math.abs(c[i]) < 1e-9 ? 0 : Number(c[i].toFixed(3));
      e.cell.classList.toggle("is-set", Math.abs(c[i]) > 1e-9);
    });
    PLANT.forEach(function (p) {
      el("pe-" + p[0]).value = S[p[0]];
      el("pe-" + p[0] + "-out").textContent = p[0] === "nic" ? S.nic : S[p[0]].toFixed(p[1]);
    });
  }
  function takeOver() { if (!S.coef) S.coef = D.coef[frame].slice(); }
  function setMode(on) {
    explore = on;
    root.classList.toggle("is-explore", on);
    mode.textContent = on ? "Back to the replay" : "Try it yourself";
    mode.setAttribute("aria-pressed", on ? "true" : "false");
    if (on) { setPlaying(false); takeOver(); }
    else { S.coef = null; S.mu = 1; S.umax = 5; S.nic = 8; S.spread = 4.2; }
    syncKnobs(); render();
  }

  fetch("static/data/policy_evolution.json?v=8").then(function (r) { return r.json(); }).then(function (d) {
    D = d;
    D.termsPlain = ["x₁", "x₂", "x₁²", "x₁x₂", "x₂²",
      "x₁³", "x₁²x₂", "x₁x₂²", "x₂³",
      "x₁⁴", "x₁³x₂", "x₁²x₂²", "x₁x₂³", "x₂⁴"];
    slider.max = D.frames - 1;
    frame = D.frames - 1;
    slider.value = frame;
    root.classList.remove("is-loading");
    buildCoefRow();
    buildLipTrace();
    buildTermLabels();
    buildEditor();
    syncKnobs();
    render();

    slider.addEventListener("input", function () {
      frame = parseInt(slider.value, 10);
      setPlaying(false);
      if (explore) { S.coef = D.coef[frame].slice(); syncKnobs(); }
      render();
    });
    play.addEventListener("click", function () {
      if (!playing && frame === D.frames - 1) { frame = 0; slider.value = 0; }
      setPlaying(!playing);
    });
    mode.addEventListener("click", function () { setMode(!explore); });
    reset.addEventListener("click", function () {
      S.coef = D.coef[frame].slice(); S.mu = 1; S.umax = 5; S.nic = 8; S.spread = 4.2;
      syncKnobs(); render();
    });
    PLANT.forEach(function (p) {
      el("pe-" + p[0]).addEventListener("input", function (ev) {
        S[p[0]] = parseFloat(ev.target.value);
        el("pe-" + p[0] + "-out").textContent = p[0] === "nic" ? S.nic : S[p[0]].toFixed(p[1]);
        render();
      });
    });

    var drag = null;
    cMap.style.cursor = "grab";
    cMap.addEventListener("pointerdown", function (ev) {
      drag = { x: ev.clientX, y: ev.clientY, az: view.azim, el: view.elev };
      cMap.setPointerCapture(ev.pointerId); cMap.style.cursor = "grabbing";
    });
    cMap.addEventListener("pointermove", function (ev) {
      if (!drag) return;
      view.azim = drag.az - (ev.clientX - drag.x) * 0.4;
      view.elev = Math.max(4, Math.min(80, drag.el + (ev.clientY - drag.y) * 0.3));
      drawSurface(coefNow());
    });
    ["pointerup", "pointercancel"].forEach(function (t) {
      cMap.addEventListener(t, function () { drag = null; cMap.style.cursor = "grab"; });
    });

    var resize;
    window.addEventListener("resize", function () {
      clearTimeout(resize); resize = setTimeout(render, 150);
    });
  }).catch(function (err) {
    root.classList.remove("is-loading");
    readout.textContent = "The training history could not be loaded.";
    if (window.console) console.error("policy-explorer:", err);
  });
})();
