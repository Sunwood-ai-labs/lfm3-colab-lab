// Validate the published HTML, including local assets, anchors and locale parity.
import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'

const root = path.resolve('docs/.vitepress/dist')
const base = '/lfm3-colab-lab/'
const origin = 'https://docs.invalid'
const pages = new Map()
const errors = []

for (const name of ['README.md', 'README.ja.md', 'CONTRIBUTING.md', 'CHANGELOG.md']) {
  const markdown = await readFile(name, 'utf8')
  const targets = [...markdown.matchAll(/\]\(([^\s)]+)(?:\s+"[^"]*")?\)/g)].map(match => match[1])
  targets.push(...[...markdown.matchAll(/\b(?:href|src)="([^"]+)"/g)].map(match => match[1]))
  for (const raw of targets) {
    const target = raw.replace(/^<|>$/g, '').split('#')[0]
    if (!target || /^[a-z]+:/i.test(target) || target.startsWith('/')) continue
    try {
      await stat(path.resolve(path.dirname(name), decodeURIComponent(target)))
    } catch {
      errors.push(`${name}: missing source link: ${target}`)
    }
  }
}

async function collect(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name)
    if (entry.isDirectory()) await collect(full)
    else if (entry.name.endsWith('.html')) {
      const html = await readFile(full, 'utf8')
      pages.set(path.relative(root, full).replaceAll('\\', '/'), {
        html,
        ids: new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1])),
      })
    }
  }
}

await collect(root)
if (!pages.size) errors.push('No generated HTML pages found')
for (const required of ['index.html', 'results.html', 'verification.html', 'python-cli.html']) {
  for (const locale of ['', 'en/']) {
    if (!pages.has(locale + required)) errors.push(`Missing guide: ${locale}${required}`)
  }
}

let links = 0
for (const [name, page] of pages) {
  const current = new URL(base + name, origin)
  for (const match of page.html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    const raw = match[1].replaceAll('&amp;', '&')
    const url = new URL(raw, current)
    if (url.origin !== origin) continue
    links++
    if (!url.pathname.startsWith(base)) {
      errors.push(`${name}: link escapes the project base: ${raw}`)
      continue
    }
    let target = decodeURIComponent(url.pathname.slice(base.length)) || 'index.html'
    if (target.endsWith('/')) target += 'index.html'
    const candidates = [target, `${target}.html`, `${target}/index.html`]
    let found
    for (const candidate of candidates) {
      try {
        if ((await stat(path.join(root, candidate))).isFile()) {
          found = candidate
          break
        }
      } catch {}
    }
    if (!found) {
      errors.push(`${name}: missing local target: ${raw}`)
      continue
    }
    target = found
    if (url.hash && pages.has(target)) {
      const id = decodeURIComponent(url.hash.slice(1))
      if (!pages.get(target).ids.has(id)) errors.push(`${name}: missing anchor: ${raw}`)
    }
  }
}

for (const error of errors) console.error(error)
console.log(`Checked ${pages.size} HTML pages and ${links} local links/assets; ${errors.length} problems`)
process.exitCode = errors.length ? 1 : 0
