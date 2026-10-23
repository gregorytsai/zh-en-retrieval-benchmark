
// Sort on source numbers, never on formatted milliseconds/seconds or rounded percentages.
const comparison = document.querySelector('#comparison');
let sortColumn = 5, sortDescending = true;
comparison?.querySelectorAll('.sort-button').forEach(button => {
  button.addEventListener('click', () => {
    const column = Number(button.dataset.column);
    sortDescending = column === sortColumn ? !sortDescending : column >= 2 && column <= 5;
    sortColumn = column;
    const body = comparison.querySelector('tbody');
    const rows = [...body.rows].map((row, index) => ({row, index, values: JSON.parse(row.dataset.sortValues)}));
    rows.sort((a, b) => {
      const x = a.values[column], y = b.values[column];
      const diff = typeof x === 'number' ? x - y : x.localeCompare(y, 'zh-Hant', {numeric: true});
      return (sortDescending ? -diff : diff) || a.index - b.index;
    });
    rows.forEach(({row}) => body.append(row));
    comparison.querySelectorAll('.sort-button').forEach(other => {
      const active = Number(other.dataset.column) === column;
      other.parentElement.setAttribute('aria-sort', active ? (sortDescending ? 'descending' : 'ascending') : 'none');
      other.querySelector('span').textContent = active ? (sortDescending ? ' ↓' : ' ↑') : ' ↕';
    });
    document.querySelector('#sort-status').textContent = `排序：${button.dataset.label}，由${sortDescending ? '高到低' : '低到高'}`;
  });
});

const chart = document.querySelector('#retrieval-bars');
const directions = {all: '全部平均', 'eng-zho': '中文查英文', 'zho-zho': '中文查中文', 'eng-eng': '英文查英文'};
let chartDirection = 'all', chartReranker = 'standalone';
function chartMetric(embedding, reranker) {
  const source = chartDirection === 'all' ? BENCHMARK.pairs : BENCHMARK.directions.filter(p => p.config === chartDirection);
  return source.find(p => p.embedding === embedding && p.reranker === reranker);
}
function element(tag, className, text) {
  const el = document.createElement(tag); el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function renderChart() {
  if (!chart) return;
  const hasReranker = chartReranker !== 'standalone';
  const rows = BENCHMARK.pairs.filter(p => p.reranker === 'standalone').map(p => ({
    embedding: p.embedding, base: chartMetric(p.embedding, 'standalone'), result: chartMetric(p.embedding, chartReranker)
  })).sort((a, b) => Number(Boolean(b.result)) - Number(Boolean(a.result)) ||
    (b.result?.top1 ?? b.base.top1) - (a.result?.top1 ?? a.base.top1));
  chart.replaceChildren();
  for (const {embedding, base, result} of rows) {
    const row = element('div', 'metric-row' + (!result ? ' untested' : ''));
    row.dataset.embedding = embedding;
    const label = element('a', 'metric-name', BENCHMARK.names[embedding]);
    label.href = `models/${embedding}.html`;
    const track = element('div', 'metric-track');
    track.setAttribute('aria-hidden', 'true');
    const baseFill = element('span', 'metric-base');
    baseFill.style.width = `${Math.min(base.top1, result?.top1 ?? base.top1) * 100}%`;
    track.append(baseFill);
    const value = element('div', 'metric-value');
    if (!result) {
      value.append(element('strong', '', '未測此配對'), element('small', '', `單模型 ${(base.top1 * 100).toFixed(2)}%`));
    } else {
      const delta = (result.top1 - base.top1) * 100;
      if (hasReranker && Math.abs(delta) > 1e-9) {
        const gain = element('span', delta >= 0 ? 'metric-gain' : 'metric-loss');
        gain.style.left = `${Math.min(base.top1, result.top1) * 100}%`;
        gain.style.width = `${Math.abs(delta)}%`; track.append(gain);
      }
      value.append(element('strong', '', `${(result.top1 * 100).toFixed(2)}%`));
      if (hasReranker) value.append(element('span', 'metric-delta' + (delta < 0 ? ' negative' : ''), `${delta > 0 ? '+' : ''}${delta.toFixed(2)} 百分點`));
      const time = result.total_ms_mean < 100 ? `${result.total_ms_mean.toFixed(2)} ms` : `${(result.total_ms_mean / 1000).toFixed(2)} 秒`;
      value.append(element('small', '', hasReranker ? `原本 ${(base.top1 * 100).toFixed(2)}% · ${time}` : `${time} · 單模型`));
    }
    row.append(label, track, value); chart.append(row);
  }
  const rerankerName = hasReranker ? BENCHMARK.names[chartReranker] : '不加重排器';
  document.querySelector('#chart-title').textContent = hasReranker ? '加上重排器，命中率提升多少？' : '不加重排器，誰先找到對的資料？';
  document.querySelector('#chart-summary').textContent = `${directions[chartDirection]} · ${rerankerName}；${chartDirection === 'all' ? '三方向等權平均，每個已測組合 300 題' : '每個已測組合 100 題'}。按${hasReranker ? '重排後' : '單模型'}命中率由高到低排列${hasReranker ? '；Granite 與 F2LLM 沒有重排測量，灰色長條僅供單模型參考' : ''}。`;
}
for (const key of ['direction', 'reranker']) {
  document.querySelectorAll(`#chart-${key} button`).forEach(button => button.addEventListener('click', () => {
    if (key === 'direction') chartDirection = button.dataset.value;
    else chartReranker = button.dataset.value;
    document.querySelectorAll(`#chart-${key} button`).forEach(other => other.setAttribute('aria-pressed', String(other === button)));
    renderChart();
  }));
}
renderChart();
