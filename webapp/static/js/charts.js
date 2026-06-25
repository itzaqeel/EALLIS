/* ══════════════════════════════════════════════════════════════════════
   EALLIS — Chart.js Helpers (v4 Dark Theme)
   ══════════════════════════════════════════════════════════════════════ */

const CHART_COLORS = {
  cyan:    'rgba(0, 200, 255, 1)',
  cyanDim: 'rgba(0, 200, 255, 0.15)',
  purple:  'rgba(157, 78, 221, 1)',
  purpleDim: 'rgba(157, 78, 221, 0.15)',
  green:   'rgba(0, 230, 118, 1)',
  greenDim:'rgba(0, 230, 118, 0.15)',
  amber:   'rgba(255, 179, 0, 1)',
  amberDim:'rgba(255, 179, 0, 0.15)',
  red:     'rgba(255, 82, 82, 1)',
  redDim:  'rgba(255, 82, 82, 0.15)',
  pink:    'rgba(240, 101, 149, 1)',
  pinkDim: 'rgba(240, 101, 149, 0.15)',
  grid:    'rgba(255, 255, 255, 0.06)',
  text:    'rgba(148, 163, 184, 1)',
};

/* ── Shared Chart.js defaults ──────────────────────────────────────── */
function setChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = CHART_COLORS.text;
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(8, 12, 20, 0.9)';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.titleFont = { weight: '600' };
}

/* ── Create gradient fill for area charts ──────────────────────────── */
function createGradient(ctx, color, height) {
  const gradient = ctx.createLinearGradient(0, 0, 0, height || 300);
  gradient.addColorStop(0, color.replace('1)', '0.25)'));
  gradient.addColorStop(1, color.replace('1)', '0.02)'));
  return gradient;
}

/* ── Shared axis configuration ─────────────────────────────────────── */
function darkScaleOptions(titleX, titleY) {
  return {
    x: {
      grid: { color: CHART_COLORS.grid, drawBorder: false },
      ticks: { color: CHART_COLORS.text, maxTicksLimit: 12 },
      title: titleX ? { display: true, text: titleX, color: CHART_COLORS.text } : undefined,
    },
    y: {
      grid: { color: CHART_COLORS.grid, drawBorder: false },
      ticks: { color: CHART_COLORS.text },
      title: titleY ? { display: true, text: titleY, color: CHART_COLORS.text } : undefined,
    },
  };
}

/* ── Build a line chart ────────────────────────────────────────────── */
function buildLineChart(canvasId, labels, datasets, xTitle, yTitle) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top' },
      },
      scales: darkScaleOptions(xTitle, yTitle),
      elements: {
        line: { tension: 0.3, borderWidth: 2 },
        point: { radius: 0, hoverRadius: 4 },
      },
    },
  });
}

/* ── Build a bar chart ─────────────────────────────────────────────── */
function buildBarChart(canvasId, labels, datasets, xTitle, yTitle) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
      },
      scales: darkScaleOptions(xTitle, yTitle),
    },
  });
}

/* ── Build a grouped bar chart (comparison) ────────────────────────── */
function buildComparisonChart(canvasId, labels, modelsData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const colors = [
    { bg: CHART_COLORS.amberDim, border: CHART_COLORS.amber },
    { bg: CHART_COLORS.purpleDim, border: CHART_COLORS.purple },
    { bg: CHART_COLORS.cyanDim, border: CHART_COLORS.cyan },
  ];

  const datasets = modelsData.map((m, i) => ({
    label: m.name,
    data: m.values,
    backgroundColor: colors[i].bg,
    borderColor: colors[i].border,
    borderWidth: 1.5,
    borderRadius: 6,
    barPercentage: 0.7,
  }));

  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%`
          }
        }
      },
      scales: {
        ...darkScaleOptions(null, 'mAP (%)'),
        y: {
          ...darkScaleOptions().y,
          beginAtZero: true,
          max: 60,
          title: { display: true, text: 'mAP (%)', color: CHART_COLORS.text },
        },
      },
    },
  });
}

/* ── Animated number counter ───────────────────────────────────────── */
function animateNumbers() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 1;
    const duration = 1200;
    const start = performance.now();

    function update(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;
      el.textContent = current.toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(update);
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          requestAnimationFrame(update);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });

    observer.observe(el);
  });
}

/* ── Init on load ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setChartDefaults();
  animateNumbers();
});
