(function () {
  var scaleBusy = !!(
    document.getElementById("pool-scale-log") ||
    document.getElementById("pool-scale-starting")
  );
  var buildBusy = !!document.getElementById("build-job-active");

  function setDisabled(el, disabled, titleWhenDisabled) {
    if (!el) {
      return;
    }
    el.disabled = !!disabled;
    if (disabled && titleWhenDisabled) {
      el.title = titleWhenDisabled;
    } else {
      el.removeAttribute("title");
    }
  }

  function applyScaleUi(data) {
    var busy = !!(data.pool_scale_job || data.pool_scale_starting);
    setDisabled(
      document.getElementById("scale-capacity-btn"),
      busy,
      busy
        ? "A scale job is already running — use Stop scale first"
        : ""
    );
    setDisabled(document.getElementById("stop-scale-btn"), !data.pool_scale_job);
    var status = document.getElementById("pool-scale-status");
    if (status && data.pool_scale_job) {
      var job = data.pool_scale_job;
      var id = String(job.id || "").slice(0, 8);
      status.textContent =
        "Scaling job " + id + " since " + (job.started_at || "") +
        " (pid " + (job.pid || "?") + ").";
    }
    var poolLog = document.getElementById("pool-scale-log");
    if (poolLog && data.pool_scale_log_tail) {
      poolLog.textContent = data.pool_scale_log_tail;
    }
  }

  function applyPauseUi(paused) {
    var resume = document.getElementById("ingest-resume-form");
    var pause = document.getElementById("ingest-pause-form");
    var label = document.getElementById("ingest-pause-label");
    if (resume) {
      resume.hidden = !paused;
    }
    if (pause) {
      pause.hidden = !!paused;
    }
    if (label) {
      label.innerHTML = paused
        ? "<strong>paused</strong>"
        : "<strong>running</strong>";
    }
  }

  function poll() {
    fetch("/api/settings/status", { credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) {
          return null;
        }
        return response.json();
      })
      .then(function (data) {
        if (!data) {
          return;
        }
        var buildLog = document.getElementById("build-log");
        if (buildLog && data.log_tail) {
          buildLog.textContent = data.log_tail;
        }

        applyScaleUi(data);
        if (typeof data.ingest_paused === "boolean") {
          applyPauseUi(data.ingest_paused);
        }

        var scaleNow = !!(data.pool_scale_job || data.pool_scale_starting);
        var buildNow = !!data.build_job;

        // Job finished (or prep ended): reload so plan/history/buttons match server.
        if (scaleBusy && !scaleNow) {
          window.location.replace("/settings?tab=ingest");
          return;
        }
        if (buildBusy && !buildNow) {
          window.location.replace("/settings?tab=memgraph_build");
          return;
        }
        scaleBusy = scaleNow;
        buildBusy = buildNow;
      })
      .catch(function () {});
  }

  document.querySelectorAll("form[data-busy-label]").forEach(function (form) {
    form.addEventListener("submit", function () {
      var button = form.querySelector('button[type="submit"]');
      if (!button || button.disabled) {
        return;
      }
      button.disabled = true;
      button.textContent = form.getAttribute("data-busy-label") || "Working…";
    });
  });

  if (document.querySelector("[data-settings-live]")) {
    setInterval(poll, 4000);
    poll();
  }
})();
