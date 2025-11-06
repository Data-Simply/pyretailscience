# Plot Gallery

PyRetailScience provides a comprehensive set of plotting functions designed specifically for retail analytics. All plots
use a consistent API and come pre-styled with retail-friendly color schemes and professional styling.

## Plot Types

<!-- markdownlint-disable MD033 -->
<style>
/* Matplotlib-style gallery container */
.sphx-glr-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, 200px);
  gap: 15px;
  margin: 20px 0;
  justify-content: start;
}

/* Individual thumbnail container - consistent card style */
.sphx-glr-thumbcontainer {
  text-decoration: none;
  color: inherit;
  display: block;
  text-align: center;
  transition: transform 0.2s ease;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 12px;
  background: var(--md-default-bg-color, #fff);
  width: 200px;
}

.sphx-glr-thumbcontainer:hover {
  transform: scale(1.02);
  text-decoration: none;
  border-color: #007acc;
  box-shadow: 0 2px 8px rgba(0, 122, 204, 0.15);
}

/* Card header - plot type name */
.sphx-glr-thumb-title {
  font-size: 14px;
  margin-bottom: 12px;
  font-weight: 600;
  color: var(--md-default-fg-color, #333);
  line-height: 1.2;
  border-bottom: 1px solid var(--md-default-fg-color--lightest, #eee);
  padding-bottom: 8px;
}

/* Thumbnail image */
.sphx-glr-thumb img {
  width: 160px;
  height: 112px;
  object-fit: contain;
  border: 1px solid #ddd;
  border-radius: 4px;
  display: block;
  margin: 0 auto;
  background: #fff;
  padding: 4px;
}

/* Remove image-specific hover effects - only hover on outer container */

/* Coming soon placeholder */
.coming-soon-thumb {
  width: 160px;
  height: 112px;
  background: #f0f0f0;
  border: 1px dashed #ccc;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: #999;
  margin: 0 auto;
}

.coming-soon-container {
  color: #666;
  cursor: default;
  border: 1px solid #ddd;
  background: var(--md-default-bg-color, #f9f9f9);
}

.coming-soon-container:hover {
  transform: none;
  border-color: #ddd;
  box-shadow: none;
}

.coming-soon-title {
  font-size: 14px;
  margin-bottom: 12px;
  font-weight: 600;
  color: var(--md-default-fg-color--light, #666);
  line-height: 1.2;
  border-bottom: 1px solid var(--md-default-fg-color--lightest, #ddd);
  padding-bottom: 8px;
}
</style>

<div class="sphx-glr-gallery">
  <a href="plots/area/" class="sphx-glr-thumbcontainer">
    <div class="sphx-glr-thumb-title">Area Plot</div>
    <div class="sphx-glr-thumb">
      <img src="../assets/gallery/area_thumbnail.png" alt="Area Plot">
    </div>
  </a>
</div>

### Coming Soon

<div class="sphx-glr-gallery">
  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Time Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Waterfall Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Cohort Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Venn Diagram</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Period on Period Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Broken Timeline Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Index Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>

  <div class="sphx-glr-thumbcontainer coming-soon-container">
    <div class="coming-soon-title">Price Plot</div>
    <div class="coming-soon-thumb">Preview</div>
  </div>
</div>

## Getting Started

Each plot page includes:

- **Basic examples** - Simple usage to get started quickly
- **Configuration options** - Major built-in features demonstrated with code
- **Realistic data** - Retail domain examples you can copy and adapt
- **Best practices** - Tips for effective visualization

All examples use the consistent PyRetailScience import pattern:

```python
from pyretailscience.plots import area
# Then call: area.plot(...)
```

## Additional Resources

- **[API Reference](../api/plots/area.md)** - Complete parameter documentation for all plots
- **[Examples](../examples/retention.ipynb)** - End-to-end analysis workflows combining multiple plots
- **[Styling Guide](../api/plots/styles/tailwind.md)** - Customize colors, themes, and appearance
