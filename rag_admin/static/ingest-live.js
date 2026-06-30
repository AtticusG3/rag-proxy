(function () {
  "use strict";

  var POLL_ACTIVE_MS = 8000;
  var POLL_IDLE_MS = 30000;
  var WINDOW_5_MS = 5 * 60 * 1000;
  var WINDOW_15_MS = 15 * 60 * 1000;
  var MIN_ELAPSED_MS = 4800;

  var tbody = document.getElementById("ingest-files-body");
  if (!tbody) {
    return;
  }

  var velocityEl = document.getElementById("ingest-velocity");
  var liveBadge = document.getElementById("ingest-live-badge");
  var chunkSamples = [];
  var timerId = null;

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function statusPill(status) {
    return (
      '<span class="pill pill--' +
      escapeHtml(String(status).replace(/ /g, "-")) +
      '">' +
      escapeHtml(status) +
      "</span>"
    );
  }

  function formatDt(value) {
    if (!value) {
      return "";
    }
    var dt = new Date(value);
    if (Number.isNaN(dt.getTime())) {
      return escapeHtml(value);
    }
    var pad = function (n) {
      return String(n).padStart(2, "0");
    };
    var months = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    return (
      pad(dt.getDate()) +
      "-" +
      months[dt.getMonth()] +
      "-" +
      dt.getFullYear() +
      " " +
      pad(dt.getHours()) +
      ":" +
      pad(dt.getMinutes()) +
      ":" +
      pad(dt.getSeconds())
    );
  }

  function actionCell(file) {
    var parts = [];
    if (file.status === "failed" || file.is_stalled) {
      var label = file.is_stalled ? "Restart" : "Retry";
      parts.push(
        '<form method="post" action="/api/ingest/retry-form" class="inline">' +
          '<input type="hidden" name="file_path" value="' +
          escapeHtml(file.file_path) +
          '">' +
          '<button type="submit" class="btn btn--ghost btn--small">' +
          label +
          "</button></form>"
      );
    }
    parts.push(
      '<form method="post" action="/api/ingest/dismiss-form" class="inline">' +
        '<input type="hidden" name="file_path" value="' +
        escapeHtml(file.file_path) +
        '">' +
        '<button type="submit" class="btn btn--ghost btn--small">Remove</button></form>'
    );
    return '<td class="actions-cell">' + parts.join("") + "</td>";
  }

  function renderRows(files) {
    if (!files.length) {
      tbody.innerHTML =
        '<tr><td colspan="6"><div class="empty-state">' +
        '<p class="empty-state__title">No ingest state</p>' +
        '<p class="empty-state__body">Subscribe in Explorer or scan storage after adding files.</p>' +
        "</div></td></tr>";
      return;
    }

    tbody.innerHTML = files
      .map(function (file) {
        return (
          "<tr>" +
          '<td><span class="file-chip">' +
          escapeHtml(file.file_name || "") +
          "</span>" +
          (file.file_missing
            ? ' <span class="pill pill--failed" title="File no longer on disk">missing</span>'
            : "") +
          "</td>" +
          "<td>" +
          statusPill(file.display_status || file.status || "pending") +
          "</td>" +
          '<td class="mono">' +
          escapeHtml(String(file.chunks_embedded || 0)) +
          "</td>" +
          '<td class="error">' +
          escapeHtml(file.last_error || "") +
          "</td>" +
          '<td class="faint">' +
          formatDt(file.updated_at) +
          "</td>" +
          actionCell(file) +
          "</tr>"
        );
      })
      .join("");
  }

  function recordChunkSample(total, now) {
    chunkSamples.push({ t: now, total: total });
    var cutoff = now - WINDOW_15_MS;
    while (chunkSamples.length > 1 && chunkSamples[0].t < cutoff) {
      chunkSamples.shift();
    }
  }

  function rateBetween(baseline, current) {
    var elapsedMs = current.t - baseline.t;
    if (elapsedMs < MIN_ELAPSED_MS) {
      return null;
    }
    var delta = current.total - baseline.total;
    if (delta < 0) {
      return null;
    }
    return Math.round(delta / (elapsedMs / 60000));
  }

  function baselineForWindow(samples, windowMs) {
    if (!samples.length) {
      return null;
    }
    var current = samples[samples.length - 1];
    var targetT = current.t - windowMs;
    var baseline = samples[0];
    for (var i = 0; i < samples.length; i++) {
      if (samples[i].t <= targetT) {
        baseline = samples[i];
      } else {
        break;
      }
    }
    return baseline;
  }

  function rateOverWindow(samples, windowMs) {
    if (samples.length < 2) {
      return null;
    }
    var current = samples[samples.length - 1];
    var baseline = baselineForWindow(samples, windowMs);
    if (!baseline || baseline.t === current.t) {
      return null;
    }
    return rateBetween(baseline, current);
  }

  function formatRate(value) {
    if (value === null) {
      return "measuring...";
    }
    return value + " chunks/min";
  }

  function updateVelocity(stats) {
    if (!velocityEl) {
      return;
    }

    var total = stats.total_chunks || 0;
    var active = stats.active || 0;
    var now = Date.now();

    if (active > 0) {
      recordChunkSample(total, now);
      var nowRate = null;
      if (chunkSamples.length >= 2) {
        nowRate = rateBetween(
          chunkSamples[chunkSamples.length - 2],
          chunkSamples[chunkSamples.length - 1]
        );
      }
      var rate5m = rateOverWindow(chunkSamples, WINDOW_5_MS);
      var rate15m = rateOverWindow(chunkSamples, WINDOW_15_MS);
      velocityEl.textContent =
        active +
        " file(s) in queue · " +
        total.toLocaleString() +
        " chunks embedded · now " +
        formatRate(nowRate) +
        " · 5m " +
        formatRate(rate5m) +
        " · 15m " +
        formatRate(rate15m);
      velocityEl.hidden = false;
    } else {
      chunkSamples = [];
      velocityEl.textContent =
        (stats.indexed || 0) +
        " indexed · " +
        total.toLocaleString() +
        " total chunks";
      velocityEl.hidden = false;
    }

    if (liveBadge) {
      liveBadge.hidden = active === 0;
      liveBadge.textContent = active > 0 ? "Live" : "";
    }
  }

  function scheduleNext(active) {
    if (timerId) {
      window.clearTimeout(timerId);
    }
    timerId = window.setTimeout(refresh, active > 0 ? POLL_ACTIVE_MS : POLL_IDLE_MS);
  }

  function refresh() {
    fetch("/api/ingest/status", { credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("status " + response.status);
        }
        return response.json();
      })
      .then(function (payload) {
        var files = payload.files || [];
        var stats = payload.stats || {};
        renderRows(files);
        updateVelocity(stats);
        scheduleNext(stats.active || 0);
      })
      .catch(function () {
        scheduleNext(1);
      });
  }

  refresh();
})();
