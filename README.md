# SD-DPC project page

Project website for **SD-DPC: Sparse Dictionary Differentiable Predictive Control**
(Ali Reza Daneshvar Garmroodi, Jan Drgoňa).

- Paper: https://arxiv.org/abs/2610.02466
- Code: https://github.com/SOLARIS-JHU/SD-DPC

## Layout

```text
index.html                 single-page site (MathJax for equations, no build step)
static/css/sd-dpc.css      the whole theme, light + dark
static/images/             method diagram, evaluation, sweeps, identification, video poster
static/videos/             Van der Pol policy-training animation
static/pdfs/               vector versions of the method and evaluation figures
```

Figures and the animation are copies of `results/` and `assets/` in the code repository.

## Preview locally

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy on GitHub Pages

Push to a repository, then enable **Settings → Pages → Deploy from a branch** (`main`, root).
`.nojekyll` is present so `static/` is served as-is. If the site is hosted under a path other
than `https://solaris-jhu.github.io/SD-DPC/`, update the absolute `og:`/`twitter:` URLs in
`index.html`.

Based on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
adopted from [Nerfies](https://nerfies.github.io). Licensed CC BY-SA 4.0.
