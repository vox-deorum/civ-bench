"""The GitHub Pages workflow written into a published report repository."""

from __future__ import annotations

WORKFLOW_PATH = ".github/workflows/pages.yml"

# Deploys the repository root as the Pages site on every push to main. The
# upload action leaves hidden entries such as .git and .github out of the site.
PAGES_WORKFLOW = """\
name: Deploy report to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v4
        with:
          path: .
      - id: deployment
        uses: actions/deploy-pages@v4
"""
