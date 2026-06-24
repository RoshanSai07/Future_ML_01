const actualCanvas = document.getElementById("actualChart");
if (actualCanvas && typeof forecastData !== "undefined") {
  const labels = forecastData.map((item) => item.Date);
  const actual = forecastData.map((item) => item.Actual);
  const ensemble = forecastData.map((item) => item.Ensemble);
  new Chart(actualCanvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Actual Revenue",
          data: actual,
          borderWidth: 3,
          pointRadius: 0,
        },
        {
          label: "Ensemble Prediction",
          data: ensemble,
          borderWidth: 3,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
    },
  });
}
const dailyCanvas = document.getElementById("dailyRevenueChart");
if (dailyCanvas && typeof dailyData !== "undefined") {
  const labels = dailyData.map((item) => item.InvoiceDate);
  const revenue = dailyData.map((item) => item.Revenue);
  new Chart(dailyCanvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Daily Revenue",
          data: revenue,
          borderWidth: 2,
          pointRadius: 0,
        },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false },
  });
}
const monthlyCanvas = document.getElementById("monthlyRevenueChart");
if (monthlyCanvas && typeof monthlyData !== "undefined") {
  const labels = monthlyData.map((item) => item.InvoiceDate);
  const revenue = monthlyData.map((item) => item.Revenue);
  new Chart(monthlyCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Monthly Revenue", data: revenue, borderWidth: 1 }],
    },
    options: { responsive: true, maintainAspectRatio: false },
  });
}
const countryCanvas = document.getElementById("countryChart");
if (countryCanvas && typeof countryData !== "undefined") {
  const labels = countryData.map((item) => item.Country);
  const revenue = countryData.map((item) => item.Revenue);
  new Chart(countryCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Revenue", data: revenue, borderWidth: 1 }],
    },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false },
  });
}
const featureCanvas = document.getElementById("featureChart");
if (featureCanvas && typeof featureData !== "undefined") {
  const labels = featureData.map((item) => item.Feature);
  const importance = featureData.map((item) => item.Importance);
  new Chart(featureCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Importance", data: importance, borderWidth: 1 }],
    },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false },
  });
}
document.querySelectorAll(".card").forEach((card) => {
  card.addEventListener("mouseenter", () => {
    card.style.transform = "translateY(-4px)";
  });
  card.addEventListener("mouseleave", () => {
    card.style.transform = "translateY(0)";
  });
});
console.log("Dashboard Loaded");
