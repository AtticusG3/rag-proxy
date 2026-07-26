(function () {
  var params = new URLSearchParams(window.location.search);
  var mode = params.get("mode") === "advanced" ? "advanced" : "basic";
  var tab = params.get("tab") || "ingest";

  function settingsUrl(nextTab, nextMode) {
    return (
      "/settings?tab=" +
      encodeURIComponent(nextTab || tab) +
      "&mode=" +
      encodeURIComponent(nextMode || mode)
    );
  }

  var scaleBusy = !!(
    document.getElementById("pool-scale-log") ||
    document.getElementById("pool-scale-starting")
  );
  var buildBusy = !!document.getElementById("build-job-active");
  var migrateBusy = !!(
    document.getElementById("migrate-log") ||
    document.getElementById("migrate-job-status")
  );

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
      busy ? "A scale job is already running — use Stop scale first" : ""
    );
    setDisabled(document.getElementById("stop-scale-btn"), !data.pool_scale_job);
    var status = document.getElementById("pool-scale-status");
    if (status && data.pool_scale_job) {
      var job = data.pool_scale_job;
      var id = String(job.id || "").slice(0, 8);
      status.textContent =
        "Scaling job " +
        id +
        " since " +
        (job.started_at || "") +
        " (pid " +
        (job.pid || "?") +
        ").";
    }
    var poolLog = document.getElementById("pool-scale-log");
    if (poolLog && data.pool_scale_log_tail) {
      poolLog.textContent = data.pool_scale_log_tail;
    }
  }

  function applyMigrateUi(data) {
    var busy = !!data.migrate_job;
    setDisabled(
      document.getElementById("sidecar-migrate-btn"),
      busy,
      busy ? "Migration already running" : ""
    );
    setDisabled(document.getElementById("sidecar-migrate-stop-btn"), !busy);
    var status = document.getElementById("migrate-job-status");
    if (status && data.migrate_job) {
      var job = data.migrate_job;
      status.textContent =
        "Migration job " +
        String(job.id || "").slice(0, 8) +
        " since " +
        (job.started_at || "") +
        " (pid " +
        (job.pid || "?") +
        ").";
    }
    var migrateLog = document.getElementById("migrate-log");
    if (migrateLog && data.migrate_log_tail) {
      migrateLog.textContent = data.migrate_log_tail;
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
        applyMigrateUi(data);
        if (typeof data.ingest_paused === "boolean") {
          applyPauseUi(data.ingest_paused);
        }

        var scaleNow = !!(data.pool_scale_job || data.pool_scale_starting);
        var buildNow = !!data.build_job;
        var migrateNow = !!data.migrate_job;

        if (scaleBusy && !scaleNow) {
          window.location.replace(settingsUrl("ingest"));
          return;
        }
        if (migrateBusy && !migrateNow) {
          window.location.replace(settingsUrl("ingest"));
          return;
        }
        if (buildBusy && !buildNow) {
          window.location.replace(settingsUrl("memgraph_build"));
          return;
        }
        scaleBusy = scaleNow;
        buildBusy = buildNow;
        migrateBusy = migrateNow;
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

  document.querySelectorAll("[data-copy-key]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var key = btn.getAttribute("data-copy-key") || "";
      if (!key || !navigator.clipboard) {
        return;
      }
      navigator.clipboard.writeText(key).then(function () {
        var prev = btn.textContent;
        btn.textContent = "copied";
        setTimeout(function () {
          btn.textContent = prev;
        }, 900);
      });
    });
  });

  function truthy(raw) {
    return ["1", "true", "yes", "on"].indexOf(String(raw || "").trim().toLowerCase()) >= 0;
  }

  function nonempty(raw) {
    return String(raw || "").trim().length > 0;
  }

  function loadControlPlane() {
    var el = document.getElementById("settings-control-plane");
    if (!el) {
      return {
        values: {},
        labels: {},
        requires: {},
        requires_nonempty: {},
        value_requires_nonempty: {},
      };
    }
    try {
      return JSON.parse(el.textContent || "{}");
    } catch (err) {
      return {
        values: {},
        labels: {},
        requires: {},
        requires_nonempty: {},
        value_requires_nonempty: {},
      };
    }
  }

  var plane = loadControlPlane();
  var form = document.getElementById("settings-form");
  var applyBar = document.getElementById("settings-apply");
  var dirtyHint = document.getElementById("settings-dirty-hint");

  function labelOf(key) {
    return plane.labels[key] || key;
  }

  function readFormValue(key) {
    if (!form) {
      return plane.values[key] || "";
    }
    var boolHidden = form.querySelector('[data-bool-value="' + key + '"]');
    if (boolHidden) {
      return boolHidden.value;
    }
    var named = form.elements.namedItem(key);
    if (!named) {
      return plane.values[key] || "";
    }
    if (named instanceof RadioNodeList) {
      return named.value || "";
    }
    return named.value || "";
  }

  function currentValues() {
    var values = Object.assign({}, plane.values);
    if (!form) {
      return values;
    }
    form.querySelectorAll("[data-setting-key]").forEach(function (field) {
      var key = field.getAttribute("data-setting-key");
      if (key) {
        values[key] = readFormValue(key);
      }
    });
    return values;
  }

  function setBool(key, on, opts) {
    opts = opts || {};
    var hidden = form && form.querySelector('[data-bool-value="' + key + '"]');
    var sw = form && form.querySelector('[data-bool-switch="' + key + '"]');
    if (!hidden || !sw) {
      return false;
    }
    var next = on ? "true" : "false";
    if (hidden.value === next && !opts.force) {
      return false;
    }
    hidden.value = next;
    sw.classList.toggle("is-on", !!on);
    sw.setAttribute("aria-checked", on ? "true" : "false");
    var state = sw.querySelector("[data-switch-state]");
    if (state) {
      state.textContent = on ? "On" : "Off";
    }
    plane.values[key] = next;
    if (!opts.silent) {
      form.dispatchEvent(new Event("change", { bubbles: true }));
    }
    return true;
  }

  function ensureRequires(key) {
    var parents = plane.requires[key] || [];
    var changed = false;
    parents.forEach(function (parent) {
      var onForm = form && form.querySelector('[data-bool-switch="' + parent + '"]');
      if (onForm && !truthy(readFormValue(parent))) {
        setBool(parent, true, { silent: true });
        changed = true;
        ensureRequires(parent);
      }
    });
    return changed;
  }

  function evaluateWarnings(values) {
    var warnings = [];
    Object.keys(plane.requires || {}).forEach(function (key) {
      if (!truthy(values[key])) {
        return;
      }
      var missing = (plane.requires[key] || []).filter(function (parent) {
        return !truthy(values[parent]);
      });
      if (!missing.length) {
        return;
      }
      warnings.push({
        key: key,
        message:
          labelOf(key) +
          " is on, but requires " +
          missing.map(labelOf).join(", ") +
          ".",
      });
    });
    Object.keys(plane.requires_nonempty || {}).forEach(function (key) {
      if (!truthy(values[key])) {
        return;
      }
      var missing = (plane.requires_nonempty[key] || []).filter(function (dep) {
        return !nonempty(values[dep]);
      });
      if (!missing.length) {
        return;
      }
      warnings.push({
        key: key,
        message:
          labelOf(key) +
          " is on, but " +
          missing.map(labelOf).join(", ") +
          " is empty.",
      });
    });
    Object.keys(plane.value_requires_nonempty || {}).forEach(function (key) {
      var current = String(values[key] || "").trim();
      var needed = (plane.value_requires_nonempty[key] || {})[current];
      if (!needed) {
        return;
      }
      var missing = needed.filter(function (dep) {
        return !nonempty(values[dep]);
      });
      if (!missing.length) {
        return;
      }
      warnings.push({
        key: key,
        message:
          labelOf(key) +
          " is set to " +
          current +
          ", but " +
          missing.map(labelOf).join(", ") +
          " is empty.",
      });
    });
    return warnings;
  }

  function renderWarnings() {
    var values = currentValues();
    var warnings = evaluateWarnings(values);
    var box = document.getElementById("settings-config-warnings");
    var list = document.getElementById("settings-config-warnings-list");
    if (box && list) {
      list.innerHTML = "";
      warnings.forEach(function (warn) {
        var li = document.createElement("li");
        li.setAttribute("data-warn-key", warn.key);
        li.textContent = warn.message;
        list.appendChild(li);
      });
      box.hidden = warnings.length === 0;
    }

    var byKey = {};
    warnings.forEach(function (warn) {
      if (!byKey[warn.key]) {
        byKey[warn.key] = [];
      }
      byKey[warn.key].push(warn.message);
    });

    document.querySelectorAll("[data-setting-key]").forEach(function (field) {
      var key = field.getAttribute("data-setting-key");
      var warnEl = field.querySelector("[data-field-warn]");
      var messages = byKey[key] || [];
      field.classList.toggle("has-warn", messages.length > 0);
      if (!warnEl) {
        return;
      }
      if (!messages.length) {
        warnEl.hidden = true;
        warnEl.textContent = "";
        return;
      }
      warnEl.hidden = false;
      warnEl.textContent = messages.join(" ");
    });

    // Soft dim dependents whose parents are off (still toggleable).
    Object.keys(plane.requires || {}).forEach(function (key) {
      var field = document.querySelector('[data-setting-key="' + key + '"]');
      if (!field) {
        return;
      }
      var parents = plane.requires[key] || [];
      var blocked = parents.some(function (parent) {
        return !truthy(values[parent]);
      });
      field.classList.toggle("is-blocked", blocked && !truthy(values[key]));
    });
  }

  if (form) {
    form.querySelectorAll("[data-bool-switch]").forEach(function (sw) {
      sw.addEventListener("click", function () {
        var key = sw.getAttribute("data-bool-switch");
        if (!key) {
          return;
        }
        var next = !truthy(readFormValue(key));
        if (next) {
          ensureRequires(key);
        }
        setBool(key, next);
        renderWarnings();
      });
    });

    form.addEventListener("input", renderWarnings);
    form.addEventListener("change", renderWarnings);
    renderWarnings();
  }

  if (form && form.hasAttribute("data-dirty-track") && applyBar) {
    var initial = new FormData(form);
    function isDirty() {
      var current = new FormData(form);
      for (var pair of current.entries()) {
        if (String(initial.get(pair[0]) || "") !== String(pair[1] || "")) {
          return true;
        }
      }
      return false;
    }
    function syncDirty() {
      var dirty = isDirty();
      applyBar.classList.toggle("is-dirty", dirty);
      if (dirtyHint) {
        dirtyHint.hidden = !dirty;
      }
    }
    form.addEventListener("input", syncDirty);
    form.addEventListener("change", syncDirty);
  }

  if (document.querySelector("[data-settings-live]")) {
    setInterval(poll, 4000);
    poll();
  }
})();
