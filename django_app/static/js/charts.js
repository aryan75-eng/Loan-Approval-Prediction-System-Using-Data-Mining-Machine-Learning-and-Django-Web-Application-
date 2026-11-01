/**
 * charts.js — Shared Chart.js helper utilities.
 * Imported on pages that render dynamic Chart.js charts.
 */

"use strict";

/**
 * Create a standard doughnut chart.
 * @param {string} canvasId - The canvas element ID
 * @param {string[]} labels
 * @param {number[]} data
 * @param {string[]} colors
 */
function createDoughnutChart(canvasId, labels, data, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderWidth: 2, borderColor: "#fff" }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
    },
  });
}

/**
 * Create a horizontal bar chart.
 * @param {string} canvasId
 * @param {string[]} labels
 * @param {number[]} data
 * @param {string} color
 */
function createHBarChart(canvasId, labels, data, color = "#1976D2") {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data, backgroundColor: color, borderRadius: 5 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    },
  });
}

/**
 * Destroy all Chart.js instances attached to a page (useful for SPA-style updates).
 */
function destroyAllCharts() {
  Chart.helpers.each(Chart.instances, (instance) => instance.destroy());
}
