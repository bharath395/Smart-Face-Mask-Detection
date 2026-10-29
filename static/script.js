// Dashboard: update mask status from Flask backend
const statusEl = document.getElementById("mask-status");

if (statusEl) {
  setInterval(() => {
    fetch("/get_status")
      .then(res => res.json())
      .then(data => {
        if (data) {
          statusEl.textContent = `${data.name} | ${data.status}`;
          if (data.status === "Mask") {
            statusEl.style.color = "lightgreen";
          } else if (data.status === "No Mask") {
            statusEl.style.color = "red";
          } else {
            statusEl.style.color = "white";
          }
        }
      })
      .catch(err => console.error("Status fetch error:", err));
  }, 2000); // refresh every 2s
}

// Alerts page: load alerts dynamically
function loadAlerts() {
  const list = document.getElementById("alerts-list");
  if (!list) return;

  fetch("/get_alerts")
    .then(res => res.json())
    .then(data => {
      list.innerHTML = "";
      if (data.length === 0) {
        list.innerHTML = "<p>No alerts yet ✅</p>";
        return;
      }
      data.slice().reverse().forEach(alert => {
        list.innerHTML += `
          <div class="alert-card">
            <img src="/static/${alert.image}" alt="Alert Screenshot" />
            <div class="alert-info">
              <p><strong>Name:</strong> ${alert.name}</p>
              <p><strong>Year:</strong> ${alert.year}</p>
              <p><strong>Department:</strong> ${alert.dept}</p>
              <p><strong>Status:</strong> ${alert.status}</p>
              <p><strong>Time:</strong> ${alert.time}</p>
            </div>
          </div>
        `;
      });
    })
    .catch(err => {
      console.error("Error loading alerts:", err);
      list.innerHTML = "<p>Error loading alerts.</p>";
    });
}

// Load alerts initially if on alerts page
if (document.getElementById("alerts-list")) {
  loadAlerts();
  setInterval(loadAlerts, 5000); // refresh every 5s
}
