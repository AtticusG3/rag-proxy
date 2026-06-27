(function () {
  var logEl = document.getElementById("build-log");
  if (!logEl) {
    return;
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
        if (data.log_tail) {
          logEl.textContent = data.log_tail;
        }
      })
      .catch(function () {});
  }

  setInterval(poll, 4000);
  poll();
})();
