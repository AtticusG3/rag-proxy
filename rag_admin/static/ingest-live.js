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
  var lastTotal = null;
  var lastTime = null;
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

  function updateVelocity(stats, files) {
    if (!velocityEl) {
      return;
    }

    var total = stats.total_chunks || 0;
    var active = stats.active || 0;
    var now = Date.now();
    var rateText = "idle";

    if (active > 0) {
      if (lastTotal !== null && lastTime !== null) {
        var minutes = (now - lastTime) / 60000;
        if (minutes >= 0.08) {
          var delta = total - lastTotal;
          if (delta >= 0) {
            rateText = Math.round(delta / minutes) + " chunks/min";
          }
        }
      }
      velocityEl.textContent =
        active +
        " file(s) in queue · " +
        total.toLocaleString() +
        " chunks embedded · " +
        rateText;
      velocityEl.hidden = false;
    } else {
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

    lastTotal = total;
    lastTime = now;
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
        updateVelocity(stats, files);
        scheduleNext(stats.active || 0);
      })
      .catch(function () {
        scheduleNext(1);
      });
  }

  refresh();
})();
