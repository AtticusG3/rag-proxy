(function () {
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
        var poolLog = document.getElementById("pool-scale-log");
        if (poolLog && data.pool_scale_log_tail) {
          poolLog.textContent = data.pool_scale_log_tail;
        }
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

  if (
    document.getElementById("build-log") ||
    document.getElementById("pool-scale-log")
  ) {
    setInterval(poll, 4000);
    poll();
  }
})();
