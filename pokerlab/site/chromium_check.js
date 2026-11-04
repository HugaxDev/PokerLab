// Vérification du site dans Chromium (pokerlab.site.check, tests et smoke) :
// erreurs console, requêtes hors du site, débordement horizontal à 390 px, replay des mains.
// Usage : node pokerlab/site/chromium_check.js <url_base> <page1,page2,...> [dossier_captures]
const { chromium } = require('playwright');

(async () => {
  const [base, list, shots] = process.argv.slice(2);
  const pages = list.split(',');
  const origin = base.startsWith('http') ? new URL(base).origin : 'file://';
  const browser = await chromium.launch();
  const out = [];
  for (const scheme of ['light', 'dark']) {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: scheme, deviceScaleFactor: 1 });
    for (const name of pages) {
      const page = await ctx.newPage();
      const errors = [], external = [];
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      page.on('pageerror', e => errors.push(String(e)));
      page.on('request', r => { const u = r.url(); if (!u.startsWith(origin) && !u.startsWith('data:') && !u.startsWith('blob:')) external.push(u); });
      await page.goto(base.replace(/\/$/, '') + '/' + name, { waitUntil: 'load' });
      await page.waitForTimeout(700);
      const res = { page: name, scheme, errors, external };
      res.figures = await page.evaluate(() => document.querySelectorAll('script.figdata').length);
      res.rendered = await page.evaluate(() => document.querySelectorAll('.js-plotly-plot').length);
      if (name.startsWith('mains')) {
        const rows = await page.$$('#hands tbody tr');
        res.rows = rows.length;
        if (rows.length) {
          await rows[0].click();
          await page.waitForTimeout(300);
          res.replay = await page.evaluate(() => getComputedStyle(document.querySelector('#replay')).display !== 'none');
          const before = await page.textContent('#step');
          await page.click('#next');
          res.next = (await page.textContent('#step')) !== before;
          res.seats = await page.evaluate(() => document.querySelectorAll('#felt .seat').length);
        }
      }
      res.overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      if (shots) await page.screenshot({ path: `${shots}/${name.replace(/[^a-z0-9]/gi, '_')}_${scheme}.png`, fullPage: true });
      out.push(res);
      await page.close();
    }
    await ctx.close();
  }
  await browser.close();
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
