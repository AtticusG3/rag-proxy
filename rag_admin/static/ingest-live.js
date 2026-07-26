(function () {
  "use strict";

  var POLL_ACTIVE_MS = 8000;
  var POLL_IDLE_MS = 30000;

  var tbody = document.getElementById("ingest-files-body");
  if (!tbody) {
    return;
  }

  var velocityEl = document.getElementById("ingest-velocity");
  var liveBadge = document.getElementById("ingest-live-badge");
  var sidecarGrid = document.getElementById("sidecar-status-grid");
  var timerId = null;

  var PHASE_LABELS = {
    reindexing: "Reindexing",
    pending_reindex: "Pending reindex",
    dual_write: "Dual-write",
    stopped_for_ingest: "Paused for ingest",
    down: "Down",
    off: "Off",
    unconfigured: "Not configured",
    idle: "Idle",
  };

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function currentSortParams() {
    var params = new URLSearchParams(window.location.search);
    return {
      sort: params.get("sort") || "",
      dir: params.get("dir") || "",
      query: window.location.search || "",
    };
  }

  function formatBytes(size) {
    if (size === null || size === undefined) {
      return "";
    }
    var n = Number(size);
    if (Number.isNaN(n)) {
      return "";
    }
    if (n < 1024) {
      return n + " B";
    }
    if (n < 1024 * 1024) {
      return (n / 1024).toFixed(1) + " KB";
    }
    if (n < 1024 * 1024 * 1024) {
      return (n / (1024 * 1024)).toFixed(1) + " MB";
    }
    return (n / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }

  function formatCount(value) {
    if (value === null || value === undefined) {
      return "—";
    }
    return Number(value).toLocaleString();
  }

  function priorityCell(file) {
    var sortState = currentSortParams();
    var value = file.priority || "mid";
    var options = ["high", "mid", "low"]
      .map(function (level) {
        var label = level.charAt(0).toUpperCase() + level.slice(1);
        return (
          '<option value="' +
          level +
          '"' +
          (value === level ? " selected" : "") +
          ">" +
          label +
          "</option>"
        );
      })
      .join("");
    return (
      '<td><form method="post" action="/api/ingest/priority-form" class="priority-form">' +
      '<input type="hidden" name="file_path" value="' +
      escapeHtml(file.file_path) +
      '">' +
      '<input type="hidden" name="sort" value="' +
      escapeHtml(sortState.sort) +
      '">' +
      '<input type="hidden" name="dir" value="' +
      escapeHtml(sortState.dir) +
      '">' +
      '<select name="priority" class="priority-select priority-select--' +
      escapeHtml(value) +
      '" aria-label="Priority for ' +
      escapeHtml(file.file_name || "") +
      '" onchange="this.form.submit()">' +
      options +
      "</select></form></td>"
    );
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
        '<tr><td colspan="8"><div class="empty-state">' +
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
          priorityCell(file) +
          "<td>" +
          statusPill(file.display_status || file.status || "pending") +
          "</td>" +
          '<td class="mono">' +
          escapeHtml(String(file.chunks_embedded || 0)) +
          "</td>" +
          '<td class="mono faint">' +
          escapeHtml(formatBytes(file.file_size)) +
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

  function healthLabel(row) {
    if (!row.configured) {
      return "—";
    }
    if (row.ok === true) {
      return "OK";
    }
    if (row.ok === false) {
      return "DOWN";
    }
    return "—";
  }

  function renderSidecarPanel(title, row, countKey, countLabel) {
    var phase = row.phase || "idle";
    var phaseClass = String(phase).replace(/_/g, "-");
    var phaseLabel = PHASE_LABELS[phase] || phase;
    var errorHtml = row.last_error
      ? '<div class="sidecar-panel__error"><dt>Last error</dt><dd class="error">' +
        escapeHtml(row.last_error) +
        "</dd></div>"
      : "";
    return (
      '<article class="sidecar-panel">' +
      '<header class="sidecar-panel__head">' +
      '<h3 class="sidecar-panel__title">' +
      escapeHtml(title) +
      "</h3>" +
      '<span class="pill pill--' +
      escapeHtml(phaseClass) +
      '">' +
      escapeHtml(phaseLabel) +
      "</span></header>" +
      '<dl class="config-dl config-dl--compact">' +
      "<div><dt>Mode</dt><dd class=\"mono\">" +
      escapeHtml(row.configured ? row.mode || "—" : "—") +
      "</dd></div>" +
      "<div><dt>Health</dt><dd class=\"mono\">" +
      escapeHtml(healthLabel(row)) +
      "</dd></div>" +
      "<div><dt>" +
      escapeHtml(countLabel) +
      '</dt><dd class="mono">' +
      escapeHtml(formatCount(row[countKey])) +
      "</dd></div>" +
      errorHtml +
      "</dl></article>"
    );
  }

  function renderSidecars(sidecars) {
    if (!sidecarGrid || !sidecars) {
      return;
    }
    sidecarGrid.innerHTML =
      renderSidecarPanel("BM25", sidecars.bm25 || {}, "docs", "Docs") +
      renderSidecarPanel("TurboVec", sidecars.turbovec || {}, "vectors", "Vectors");
  }

  function formatRate(rate, stats, window) {
    if (rate !== null && rate !== undefined) {
      return Number(rate).toLocaleString() + " chunks/min";
    }
    if (window === "now" && !stats.running && stats.pending > 0) {
      return "waiting";
    }
    return "—";
  }

  function buildVelocityText(stats) {
    if (stats.velocity_text) {
      return stats.velocity_text;
    }
    var active = stats.active || 0;
    if (active <= 0) {
      return (
        (stats.indexed || 0).toLocaleString() +
        " indexed · " +
        (stats.total_chunks || 0).toLocaleString() +
        " corpus chunks"
      );
    }
    var parts = [
      active + " in queue",
      (stats.total_chunks || 0).toLocaleString() + " corpus chunks",
    ];
    if (stats.running) {
      parts.push(stats.running + " embedding");
    }
    parts.push("now " + formatRate(stats.embed_rate_now, stats, "now"));
    parts.push("5m " + formatRate(stats.embed_rate_5m, stats, "5m"));
    parts.push("15m " + formatRate(stats.embed_rate_15m, stats, "15m"));
    return parts.join(" · ");
  }

  function updateVelocity(stats, sidecarsLive) {
    if (!velocityEl) {
      return;
    }
    velocityEl.textContent = buildVelocityText(stats);
    velocityEl.hidden = false;

    if (liveBadge) {
      var active = (stats.active || 0) > 0 || !!sidecarsLive;
      liveBadge.hidden = !active;
      liveBadge.textContent = active ? "Live" : "";
    }
  }

  function scheduleNext(active) {
    if (timerId) {
      window.clearTimeout(timerId);
    }
    timerId = window.setTimeout(refresh, active ? POLL_ACTIVE_MS : POLL_IDLE_MS);
  }

  function refresh() {
    fetch("/api/ingest/status" + currentSortParams().query, {
      credentials: "same-origin",
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("status " + response.status);
        }
        return response.json();
      })
      .then(function (payload) {
        var files = payload.files || [];
        var stats = payload.stats || {};
        var sidecars = payload.sidecars || {};
        var sidecarsLive = !!payload.sidecars_live;
        renderRows(files);
        renderSidecars(sidecars);
        updateVelocity(stats, sidecarsLive);
        scheduleNext((stats.active || 0) > 0 || sidecarsLive);
      })
      .catch(function () {
        scheduleNext(1);
      });
  }

  refresh();
})();
