/* Lightweight inline-symbol typesetter for the docs viewer (and any plain HTML).
 * Walks text nodes and renders sub/superscripts (H_0, z_0^0.1429, R_1/3, ft^3/s) and
 * spelled-out OR Unicode Greek (kappa_n, κ_n) using <sub>/<sup> + Unicode — NON-italic,
 * matching the calculator's label/unit style. Skips <pre> blocks and inline <code> that
 * looks like a code identifier/path (so filenames like `chessqc_5_2_runup` stay intact).
 * Shares the sub/sup grammar with driver.js: subscripts allow digit fractions (1/3),
 * decimals (0.5), and %; superscripts are exponents (decimals ok, no "/"). */
"use strict";
(function () {
  const GREEK = {
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", zeta: "ζ", eta: "η",
    theta: "θ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π",
    rho: "ρ", sigma: "σ", tau: "τ", upsilon: "υ", phi: "φ", chi: "χ", psi: "ψ", omega: "ω",
    Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ",
    Omega: "Ω", Gamma: "Γ",
  };
  const greek = (w) => GREEK[w] || GREEK[w.toLowerCase()];
  // base may start with a Latin letter or a Unicode Greek letter (so "κ_n" works in docs)
  const BASE = "[A-Za-z\\u0391-\\u03c9]+";
  const SUBC = "(?:\\([^)]+\\)|\\d+/\\d+|\\d+\\.\\d+|[A-Za-z0-9'%]+|\\*)";
  const SUPC = "(?:\\([^)]+\\)|\\d+\\.\\d+|[A-Za-z0-9']+|\\*)";
  const SS = `(?:_${SUBC}|\\^${SUPC})`;
  const TOK_SRC = `(${BASE}(?:${SS})+|(?<![A-Za-z\\\\])(?:${Object.keys(GREEK).join("|")})(?![A-Za-z]))`;
  const SYMTOK = new RegExp(`^(${BASE})((?:${SS})*)$`);
  const SUBSUP = /([_^])(\([^)]+\)|\d+\/\d+|\d+\.\d+|[A-Za-z0-9'%]+|\*)/g;
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  function tokHTML(tok) {
    const m = SYMTOK.exec(tok);
    if (!m) return greek(tok) || esc(tok);
    let out = greek(m[1]) || esc(m[1]);
    for (const g of m[2].matchAll(SUBSUP)) {
      const tag = g[1] === "_" ? "sub" : "sup";
      out += `<${tag}>${esc(g[2])}</${tag}>`;
    }
    return out;
  }
  function html(text) {
    return String(text).split(new RegExp(TOK_SRC, "g")).map((p, i) => (i % 2 ? tokHTML(p) : esc(p))).join("");
  }
  // Inline code that is a path / filename / call / multi-underscore identifier — leave alone.
  const looksCode = (s) => /[/\\]|\.py|\.md|\(\)/.test(s) || (!/\s/.test(s) && (s.match(/_/g) || []).length >= 2);
  // A fenced block is real CODE (not an equation) if it has programming/shell keywords. The
  // reference docs put equations in fenced blocks, so those should still be typeset.
  const looksCodeBlock = (s) => /\b(import|from|def|return|class|print|lambda|elif|None|True|False|pip|python|npm|sudo|git|#include)\b|\$\s|=>|;\s*\n|::/.test(s);

  function typeset(root) {
    const has = new RegExp(TOK_SRC);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        let pre = null, code = null;
        for (let p = n.parentNode; p && p !== root; p = p.parentNode) {
          if (p.tagName === "PRE" && !pre) pre = p;
          if (p.tagName === "CODE" && !code) code = p;
        }
        if (pre) { if (looksCodeBlock(pre.textContent)) return NodeFilter.FILTER_REJECT; }
        else if (code && looksCode(code.textContent)) return NodeFilter.FILTER_REJECT;
        return has.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    const nodes = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n);
    for (const node of nodes) {
      const span = document.createElement("span");
      span.innerHTML = html(node.nodeValue);
      node.parentNode.replaceChild(span, node);
    }
  }
  window.CHESSQC_SYMFMT = { html, typeset };
})();
