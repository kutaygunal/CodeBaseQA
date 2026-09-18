// Shared source-file rendering: used by the docked panel in index.html and the
// standalone undocked window (source.html). Requires hljs + the cpp language
// pack to already be loaded on the page.
function svEscapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Re-opens/closes hljs <span> tags at each line break so per-line wrapping stays
// well-formed even when a construct (e.g. a block comment) spans multiple lines.
function svSplitHighlightedHtml(html) {
  const lines = [];
  let current = '';
  const openTags = [];
  let i = 0;
  while (i < html.length) {
    if (html[i] === '<') {
      const end = html.indexOf('>', i);
      const tag = html.slice(i, end + 1);
      if (tag.startsWith('</')) openTags.pop();
      else openTags.push(tag);
      current += tag;
      i = end + 1;
    } else if (html[i] === '\n') {
      lines.push(current + '</span>'.repeat(openTags.length));
      current = openTags.join('');
      i++;
    } else {
      current += html[i];
      i++;
    }
  }
  lines.push(current + '</span>'.repeat(openTags.length));
  return lines;
}

// Renders `data` ({text, total_lines, truncated}) from GET /api/source into
// `container`, syntax-highlighted with line numbers, highlighting [startLine,
// endLine] and scrolling it into view. Returns nothing; throws on hljs failure
// only if escaping also fails (never, practically).
function svRenderSource(container, data, startLine, endLine) {
  let highlighted;
  try {
    highlighted = hljs.highlight(data.text, { language: 'cpp' }).value;
  } catch (e) {
    highlighted = svEscapeHtml(data.text);
  }
  const lines = svSplitHighlightedHtml(highlighted);
  const frag = document.createDocumentFragment();
  for (let idx = 0; idx < lines.length; idx++) {
    const n = idx + 1;
    const row = document.createElement('div');
    const isHl = startLine && endLine && n >= startLine && n <= endLine;
    row.className = 'src-line' + (isHl ? ' hl' : '');
    row.innerHTML = `<span class="ln">${n}</span><span class="code">${lines[idx] || ' '}</span>`;
    if (isHl && n === startLine) row.id = 'src-hl-start';
    frag.appendChild(row);
  }
  container.innerHTML = '';
  container.appendChild(frag);
  if (data.truncated) {
    const warn = document.createElement('div');
    warn.className = 'error';
    warn.textContent = `File truncated to ${lines.length} lines in this viewer.`;
    container.appendChild(warn);
  }
  const target = container.querySelector('#src-hl-start');
  if (target) target.scrollIntoView({ block: 'center' });
  else container.scrollTop = 0;
}

// Fetches /api/source and renders it into `container`; shows a loading/error state.
async function svLoadAndRenderSource(container, path, startLine, endLine) {
  container.innerHTML = '<div class="loading">Loading…</div>';
  try {
    const r = await fetch('/api/source?path=' + encodeURIComponent(path));
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'not found'); }
    const data = await r.json();
    svRenderSource(container, data, startLine, endLine);
  } catch (e) {
    container.innerHTML = `<div class="error">Could not load ${svEscapeHtml(path)}: ${svEscapeHtml(String(e.message || e))}</div>`;
  }
}
