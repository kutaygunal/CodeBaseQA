/* render.js — turns an answer's markdown-ish text into HTML.
 *
 * Handles paragraphs, headings, nested lists, tables, fenced code (highlight.js) and
 * ```mermaid fences (rendered lazily with Mermaid, securityLevel "strict", theme-aware).
 * Everything is HTML-escaped first: answers are LLM output and must never inject markup.
 * `path:line` citations are linkified everywhere outside code blocks.
 *
 *   renderAnswer(text, {streaming})  -> HTML string   (streaming: unterminated fences stay plain)
 *   hydrateAnswer(rootEl)            -> highlight code + render diagrams inside rootEl
 *   rerenderDiagrams()               -> re-render every diagram (call after a theme change)
 */
(function () {
  const CITE_RE = /([\w./-]+\.(?:cpp|h|hpp|cu|cuh|cc|inc)):(\d+(?:-\d+)?)/g;
  const MERMAID_SRC = 'https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.6.0/mermaid.min.js';
  const CPP_LANGS = new Set(['cpp', 'c++', 'c', 'cc', 'cxx', 'h', 'hpp', 'cuda', 'cu']);

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function escAttr(s) {
    return esc(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function inline(text) {
    let t = esc(text);
    t = t.replace(CITE_RE, '<span class="citation">$1:$2</span>');
    t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
    return t;
  }

  function renderList(items) {
    let html = '';
    const stack = [];
    for (const it of items) {
      while (stack.length && it.indent < stack[stack.length - 1].indent) {
        html += '</li></' + stack.pop().tag + '>';
      }
      if (!stack.length || it.indent > stack[stack.length - 1].indent) {
        const tag = it.ordered ? 'ol' : 'ul';
        html += '<' + tag + '><li>' + inline(it.text);
        stack.push({ indent: it.indent, tag });
      } else {
        html += '</li><li>' + inline(it.text);
      }
    }
    while (stack.length) html += '</li></' + stack.pop().tag + '>';
    return html;
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    return s.split('|').map((c) => c.trim());
  }

  function codeBlock(lang, code, closed, opts) {
    if (lang === 'mermaid') {
      if (opts.streaming || !closed) {
        return '<pre class="code"><code>' + esc(code) + '</code></pre>' +
          '<div class="diagram-note">diagram renders when the answer is complete…</div>';
      }
      return '<div class="mermaid-block" data-src="' + escAttr(code) + '">' +
        '<div class="mermaid-render"><span class="diagram-note">rendering diagram…</span></div>' +
        '<details class="mermaid-src"><summary>diagram source</summary>' +
        '<pre class="code"><code>' + esc(code) + '</code></pre></details></div>';
    }
    return '<pre class="code"><code data-lang="' + escAttr(lang) + '">' + esc(code) + '</code></pre>';
  }

  function renderAnswer(text, opts) {
    opts = opts || {};
    const lines = String(text == null ? '' : text).split('\n');
    const out = [];
    let para = [];
    const flush = () => {
      if (para.length) { out.push('<p>' + para.map(inline).join('<br>') + '</p>'); para = []; }
    };
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];

      const fm = line.match(/^\s*```\s*([\w+#-]*)\s*$/);
      if (fm) {
        flush();
        const lang = (fm[1] || '').toLowerCase();
        const buf = [];
        let closed = false;
        i++;
        while (i < lines.length) {
          if (/^\s*```\s*$/.test(lines[i])) { closed = true; i++; break; }
          buf.push(lines[i]);
          i++;
        }
        out.push(codeBlock(lang, buf.join('\n'), closed, opts));
        continue;
      }

      const hm = line.match(/^(#{1,4})\s+(.*)$/);
      if (hm) {
        flush();
        const n = Math.min(4, hm[1].length);
        out.push('<h' + n + '>' + inline(hm[2]) + '</h' + n + '>');
        i++;
        continue;
      }

      if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
        flush();
        const head = splitRow(line);
        i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
        out.push('<div class="table-wrap"><table><thead><tr>' + head.map((c) => '<th>' + inline(c) + '</th>').join('') +
          '</tr></thead><tbody>' +
          rows.map((r) => '<tr>' + r.map((c) => '<td>' + inline(c) + '</td>').join('') + '</tr>').join('') +
          '</tbody></table></div>');
        continue;
      }

      const lm = line.match(/^(\s*)([-*]|\d+[.)])\s+(.*)$/);
      if (lm) {
        flush();
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(/^(\s*)([-*]|\d+[.)])\s+(.*)$/);
          if (m) {
            items.push({ indent: m[1].length, ordered: /\d/.test(m[2]), text: m[3] });
            i++;
          } else if (items.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*```/.test(lines[i])) {
            items[items.length - 1].text += ' ' + lines[i].trim();
            i++;
          } else break;
        }
        out.push(renderList(items));
        continue;
      }

      if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { flush(); out.push('<hr>'); i++; continue; }
      if (!line.trim()) { flush(); i++; continue; }
      para.push(line);
      i++;
    }
    flush();
    return out.join('');
  }

  // ---------------------------------------------------------------- hydration

  let mermaidLoading = null;
  function loadMermaid() {
    if (window.mermaid) return Promise.resolve();
    if (!mermaidLoading) {
      mermaidLoading = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = MERMAID_SRC;
        s.onload = () => resolve();
        s.onerror = () => { mermaidLoading = null; reject(new Error('mermaid failed to load')); };
        document.head.appendChild(s);
      });
    }
    return mermaidLoading;
  }

  function isDark() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  }

  function fallback(block, note) {
    block.classList.add('failed');
    const target = block.querySelector('.mermaid-render');
    if (target) target.innerHTML = '<span class="diagram-note">' + esc(note) + '</span>';
    const d = block.querySelector('details');
    if (d) d.open = true;
  }

  let seq = 0;
  async function renderDiagrams(root, force) {
    const blocks = (root || document).querySelectorAll('.mermaid-block');
    if (!blocks.length) return;
    try { await loadMermaid(); } catch (e) {
      blocks.forEach((b) => fallback(b, 'Could not load the diagram renderer (it is fetched from cdnjs, so it needs network access).'));
      return;
    }
    const theme = isDark() ? 'dark' : 'default';
    window.mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme, flowchart: { htmlLabels: false } });
    for (const b of blocks) {
      if (b.dataset.rendered === theme && !force) continue;
      const id = 'mmd-' + (++seq);
      try {
        const { svg } = await window.mermaid.render(id, b.dataset.src || '');
        b.querySelector('.mermaid-render').innerHTML = svg;
        b.dataset.rendered = theme;
        b.classList.remove('failed');
      } catch (e) {
        const stray = document.getElementById('d' + id);
        if (stray) stray.remove();
        fallback(b, 'Diagram could not be rendered (syntax error in the generated diagram) — showing its source.');
      }
    }
  }

  function highlightCode(root) {
    if (!window.hljs) return;
    root.querySelectorAll('pre.code code[data-lang]:not([data-done])').forEach((el) => {
      const lang = el.getAttribute('data-lang') || '';
      const code = el.textContent;
      try {
        if (CPP_LANGS.has(lang) || (!lang && /[;{}]/.test(code) && /#include|::|->|std::/.test(code))) {
          el.innerHTML = window.hljs.highlight(code, { language: 'cpp' }).value;
        }
      } catch (e) { /* leave as plain text */ }
      el.setAttribute('data-done', '1');
    });
  }

  function hydrateAnswer(root) {
    if (!root) return;
    highlightCode(root);
    renderDiagrams(root, false);
  }

  window.renderAnswer = renderAnswer;
  window.hydrateAnswer = hydrateAnswer;
  window.rerenderDiagrams = () => renderDiagrams(document, true);
})();
