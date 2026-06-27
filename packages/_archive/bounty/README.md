# Animus Bounty

Autonomous bounty and task claiming module for the [Animus](https://github.com/AreteDriver/animus) exocortex.

Discovers developer bounties across multiple platforms, scores them against your skill profile, and tracks the full lifecycle from discovery to payment.

## Supported Platforms

- **Gitcoin** -- Grants and bounties
- **GitHub** -- Bounty-labeled issues, good-first-issue, hacktoberfest
- **Replit** -- Bounties (planned)
- **Bount.ing** -- Developer tasks (planned)
- **Layer3** -- Developer quests (planned)
- **OnlyDust** -- Open source contributions (planned)
- **Generic** -- Custom JSON endpoint adapter

## Install

```bash
pip install -e packages/bounty/
# With LLM support
pip install -e "packages/bounty/[llm]"
```

## CLI

```bash
animus-bounty scan           # Scan all platforms
animus-bounty status         # Tracking summary
animus-bounty match <id>     # Evaluate a bounty
animus-bounty list-bounties  # List discovered bounties
```

## Animus MCP Tools

Four tools for integration with the Animus exocortex:

| Tool | Description |
|------|-------------|
| `bounty_scan` | Scan platforms for new bounties |
| `bounty_status` | Get tracking summary |
| `bounty_match` | Evaluate a bounty against skill profile |
| `bounty_claim` | Claim a discovered bounty |

## Configuration

Config file: `~/.animus/bounty/config.yaml`

```yaml
scan_interval_hours: 4
skills:
  languages: [python, rust, typescript, solidity]
  frameworks: [fastapi, react, nextjs, bevy]
  domains: [ai, blockchain, devtools, web]
filters:
  min_reward_usd: 5.0
  max_difficulty: hard
  auto_claim_low_effort: true
  auto_claim_max_reward_usd: 50.0
```

Environment variables:
- `GITHUB_TOKEN` -- GitHub API authentication
- `GITCOIN_API_KEY` -- Gitcoin API key
- `BOUNTY_DISCORD_WEBHOOK` -- Discord alerts for high-value bounties
- `BOUNTY_DATA_DIR` -- Override data directory (default: `~/.animus/bounty/`)

## Architecture

```
animus_bounty/
├── config.py          # BountyConfig, SkillProfile, FilterConfig
├── models.py          # Bounty, BountyStatus, Platform, ClaimResult, SkillMatch
├── platforms/
│   ├── base.py        # BasePlatform abstract scanner
│   ├── gitcoin.py     # Gitcoin grants/bounties
│   ├── github.py      # GitHub bounty-labeled issues
│   └── generic.py     # Generic JSON endpoint adapter
├── matcher.py         # Keyword + LLM skill matching
├── tracker.py         # Lifecycle tracking (discover -> claim -> paid)
├── store.py           # JSONL persistence with dedup
├── tools/
│   └── bounty_tools.py  # 4 Animus MCP tools
└── cli.py             # Typer CLI
```

## Development

```bash
pip install -e "packages/bounty/[dev]"
cd packages/bounty && pytest tests/ -v
ruff check . && ruff format --check .
```

## License

MIT
