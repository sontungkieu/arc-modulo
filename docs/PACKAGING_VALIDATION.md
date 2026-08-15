# Packaging validation — 2026-08-15

## GitHub layout

| Branch | Reproduction tip | Direct upstream parent |
|---|---|---|
| `paper/null-tta` | `ad3473a85565a023f5804430883f99a1a6062e14` | `337bf73037e9f24e9f844974d3287384abb610bf` |
| `paper/das` | `01549292b9be2d5537e34d239ab1b9d8e610c769` | `2f4b2239f29ee59f80359bdfa5ed747b6d855a1e` |
| `paper/fk-steering` | `01f881c97e516333755f5360316476e48cf3afd1` | `9413005dee1e79f80fb4561a4a7ac8eec704281b` |
| `paper/fk-correctors` | `fc7ac6985cd149ffaad81c5cf5a42203f5907ac0` | `aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6` |

GitHub API confirmed that each reproduction commit has the stated upstream
commit as its first and only parent. `main` is the default branch.

## Environment and artifact checks

- `uv lock --check`: PASS for root, `envs/null-tta`, and `envs/fk-steering`.
- Python compile check: PASS for suite wrappers, Kaggle cells, report builders,
  and DAS explainer generator.
- High-confidence secret scan: no token/private-key match. The tracked upstream
  `.env.example` contains only the key `MICROMAMBA_ENV` with a placeholder.
- GitHub file-size guard: no file above 95 MB; largest tracked source artifact
  is the upstream FK Correctors checkpoint at 12,185,398 bytes.
- PDF structural checks: report 15 pages, slide deck 22 pages, DAS explainer
  5 pages.

## Fresh-clone bootstrap test

A clean `main` clone ran `scripts/bootstrap_upstreams.py` without local object
alternates. GitHub interrupted the first FK Steering transfer near EOF; the
bounded per-branch HTTP/1.1 retry recovered automatically. The final
`scripts/verify_upstreams.py` output was `ok: true`: all four worktrees were at
the expected commit and clean.

## Full local ZIP

`diffusion-smc-repro-friend-kit-20260815.zip` is a 252 MB full snapshot that
includes the four pinned source trees, unlike thin `main`. SHA-256:

```text
a558a3000e77395edbbc82cb7ab2c80b1a2e3fae368136ca1a4004f797de1cc7
```

`unzip -t` passed. A separate extraction produced 618 files / 268 MB; all
internal `SHA256SUMS` entries and all three frozen lock checks passed.
