// Independent of dataset/3D initialization, so startup failures stay visible.
(() => {
  const banner = document.createElement('aside');
  banner.setAttribute('role', 'status');
  banner.style.cssText = 'position:relative;z-index:60;padding:12px 20px;margin:0;background:#302718;color:#ffe3a6;border-bottom:1px solid #86682e;font:14px/1.5 system-ui;';
  banner.hidden = true;
  document.body.prepend(banner);
  async function refresh() {
    try {
      const response = await fetch('/service.json', {cache: 'no-store'});
      if (!response.ok) return;
      const status = await response.json();
      banner.hidden = !['preparing', 'preparation_failed', 'paper_stopped'].includes(status.state);
      banner.textContent = status.message || '';
      if (status.state === 'paper_stopped' && location.pathname !== '/training') {
        const link = document.createElement('a');
        link.href = '/training';
        link.textContent = ' Abrir entrenamiento →';
        link.style.color = 'inherit';
        banner.append(link);
      }
    } catch { /* Existing page connection indicators handle an offline server. */ }
  }
  refresh();
  setInterval(refresh, 4000);
})();
