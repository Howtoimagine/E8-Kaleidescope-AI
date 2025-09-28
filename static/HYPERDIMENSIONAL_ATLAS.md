# Hyperdimensional Atlas Frontend

A modern WebGL control room for Kaleidoscope now lives in `static/kaleidoscope_frontend.html`. The interface is written in
ES2024 JavaScript modules and streams Three.js (r161) and D3 (v7) straight from CDN, so you can explore the visuals without
setting up a build toolchain.

## Features

- **E8 Lattice + Tetrahedron Tab** – Interactive Three.js point cloud of the 240 E8 roots with a highlighted tetrahedral
  memory simplex, starfield ambience, and orbit controls.
- **Hyperdimensional Field Tab** – GLSL-driven shader mantle that breathes with emergent wave interference and holographic
  color mapping.
- **Concept Graph Tab** – D3 force-directed library illustrating lattice agents, field dynamics, and cognitive linkages.

## Quick Start (Windows PowerShell)

```powershell
cd C:\Users\helio\Desktop\kaleidoscope
python -m http.server 8000
```

Open <http://localhost:8000/static/kaleidoscope_frontend.html> in a current Chromium, Firefox, or Safari build. Ensure
WebGL2 and hardware acceleration are enabled.

### Tips

- Drag with the mouse (or trackpad) to orbit the lattice; scroll to zoom.
- Hover graph nodes to see concept metadata; drag nodes to stabilize their orbit during the simulation.
- The field tab responds to time—leave it running to watch interference patterns phase-lock.

## Assets & Dependencies

- [Three.js 0.161.0](https://threejs.org/) + OrbitControls
- [D3.js 7.x](https://d3js.org/)
- Custom GLSL shaders embedded directly in the page

All assets are fetched at runtime; no npm install is required.
