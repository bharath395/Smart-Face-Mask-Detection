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
            statusEl.style.color = "green";
          } else if (data.status === "No Mask") {
            statusEl.style.color = "red";
          } else {
            statusEl.style.color = "black";
          }
        }
      })
      .catch(err => console.error("Status fetch error:", err));
  }, 2000); // refresh every 2s
}
