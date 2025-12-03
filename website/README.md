# Prismatron Website

Informational website for the Prismatron LED Display project. Built with React, Vite, and Tailwind CSS using a retro-futurism design.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## Deployment to GitHub Pages

The site deploys to `https://<username>.github.io/prismatron/`

### First-time setup

1. Ensure your GitHub repository has Pages enabled:
   - Go to Settings > Pages
   - Set Source to "Deploy from a branch"
   - Select `gh-pages` branch (will be created on first deploy)

### Deploy

```bash
npm run deploy
```

This will:
1. Build the site to `dist/`
2. Push `dist/` contents to the `gh-pages` branch
3. GitHub will automatically serve the site

### Manual deployment

```bash
# Build
npm run build

# Deploy to gh-pages branch
npx gh-pages -d dist
```

## Project Structure

```
website/
├── src/
│   ├── components/
│   │   └── Layout.jsx      # Main layout with sidebar nav
│   ├── pages/
│   │   ├── HomePage.jsx
│   │   ├── MechanicalPage.jsx
│   │   ├── ElectricalPage.jsx
│   │   ├── ComputePage.jsx
│   │   ├── AlgorithmPage.jsx
│   │   ├── AudioPage.jsx
│   │   ├── SoftwarePage.jsx
│   │   └── SpecsPage.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── public/
│   ├── fonts/              # Retro fonts (Orbitron, Exo 2, JetBrains Mono)
│   ├── favicon.svg
│   └── .nojekyll           # Required for GitHub Pages
├── index.html
├── tailwind.config.js
├── vite.config.js
└── package.json
```

## Adding Images/Media

Place media files in `public/images/` and reference them as `/prismatron/images/filename.jpg` in your components.
