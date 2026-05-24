# Work / Personal Boundary Policy

> **Owner:** ARETE (sole operator of animus and sole subject of this policy).
> **Authored:** 2026-05-15 as Track 9 of `PERSONAL_ROADMAP.md`.
> **Status:** Active. Applies preemptively before any new employment role and continuously thereafter.

This policy establishes the operational boundaries between animus (personal exocortex, personal IP, personal tool) and any current or future employer's work environment. The principles are universal — they apply equally to the current role, any FDE / SE / consulting role that lands, contract work, and consulting engagements. Adopting the policy *before* a new role lands is the only way to make the boundary unambiguous when questions arise.

---

## Purpose

1. **IP clarity** — animus and all related portfolio infrastructure are the operator's personal property. Work product produced for an employer belongs to the employer. The two must not blur.
2. **Damage radius limit** — if a work account is compromised, work-issued hardware is wiped, or an employment relationship ends abruptly, animus is unaffected. If animus is compromised, no work data is exposed.
3. **Reference protection** — portfolio projects (animus, anchormd, memboot, drift-monitor, BenchGoblins, etc.) remain referenceable in interviews and offers without an employer being able to claim derivative interest.
4. **Offboarding hygiene** — at the end of any role, separation is mechanical, not negotiated. Nothing personal is on work hardware; nothing work-owned is in personal infrastructure.

---

## Core boundaries (the explicit lines)

1. **Animus runs on personal hardware only.** Never installed on an employer-issued laptop, server, VM, or cloud account.
2. **No employer credentials in personal infrastructure.** SSO tokens, VPN configs, work email passwords, employer cloud-account access keys, internal API tokens — none of these enter `~/.local/share/animus/secrets.env` or any other personal secret store.
3. **No personal credentials on work hardware.** Personal API keys (Anthropic, OpenAI, GitHub, PyPI, Fly, Vercel, Stripe), personal SSH keys, personal git identity — none of these get installed on work hardware.
4. **No employer data in personal memory.** Internal documents, code, customer information, meeting notes, Slack/Teams content — none of it gets indexed by memboot, stored in animus memory, or referenced in personal notes.
5. **No personal-project work on work hardware.** Even during personal time, personal portfolio work (animus, anchormd, memboot, Substack drafts, application materials, etc.) happens on personal hardware.
6. **Work-provided AI tools are work tools.** Copilot, ChatGPT Enterprise, Claude Enterprise, Cursor licenses paid by employer, internal LLM endpoints — these are work tools. Do not bridge them to animus or use them for personal portfolio work.

---

## Hardware separation

- **Personal hardware:** desktop + Mac Mini + personal phone. Animus, the portfolio repos, application materials, financial data, family content all live here.
- **Work hardware:** whatever an employer issues. Treated as ephemeral and untrusted from a personal-IP perspective — assume the employer can image, audit, or wipe it at any time.
- **Phone-remote terminal:** when accessing personal infrastructure from a phone while at work, that's still a personal-device session (phone is personal). The boundary holds because the phone belongs to the operator, not the employer.
- **Network:** prefer personal hotspot over employer WiFi when accessing personal infrastructure from the workplace. Defense-in-depth against passive monitoring of unencrypted traffic, but more importantly avoids any ambiguity about resource use.

---

## Credential separation

| Credential type | Storage | Never goes |
|---|---|---|
| Personal Anthropic / OpenAI / GitHub / PyPI / Fly / Vercel / Stripe API keys | `~/.local/share/animus/secrets.env` (chmod 400) on personal hardware | On any employer-issued device |
| Personal SSH keys | `~/.ssh/` on personal hardware | On any employer-issued device |
| Personal git identity (`james-yng79@gmail.com` for personal work; `aretedriver@users.noreply.github.com` for OSS commits) | git config global on personal hardware | In employer-owned repos |
| Employer SSO / VPN / internal API tokens | Employer-issued password manager OR employer-issued device only | In any personal secret store, animus secrets.env, or personal password manager |
| Employer git identity (issued email) | git config local in employer repos only | As git global default on any device |
| Personal communication accounts (Gmail, personal Discord, Substack) | Personal hardware + personal phone | Logged in on employer-issued devices |

Token rotation: any time a credential crosses the boundary by accident, rotate immediately. No exceptions, no "I'll fix it later."

---

## Data separation

### Memory tagging

Animus memories carry a tag indicating boundary context. Default: `personal`.

| Tag | Meaning | Allowed in animus memory |
|---|---|---|
| `personal` | Self, family, portfolio, applications, financials | Yes |
| `portfolio` | Open-source project work, public Substack content, public repos | Yes |
| `tiaid-engagement` | Anonymized methodology notes from TIAID consulting engagements | Yes, only after client-info-scrub |
| `work-context` | Information about a current or past employer | **No** — should not enter animus memory |
| `client-confidential` | Specific identifiable client data | **No** — separate per-engagement encrypted store, not in animus |

If a memory is uncertain, default to `work-context` and exclude. Better to lose a memory than to leak a boundary.

### Memory hygiene

- Quarterly: grep memboot indexes + animus memory for inadvertent work-context contamination. Common leak patterns: meeting notes auto-imported, calendar entries, email scrapes.
- Before any role-onboarding: full grep + manual review of recent memories. Anything boundary-ambiguous gets archived to encrypted offline storage, not deleted (preservation of personal record).

---

## Knowledge boundaries

### What animus knows about employer (intentional limit)

- Public information only: company name, public products, public team list, public job description, public Glassdoor data.
- Never: internal documents, customer lists, code, architecture diagrams, internal communications, salary band internals, pending litigation, M&A talks, anything an NDA would cover.

### What animus knows about personal pursuit (no limit)

- Portfolio projects, applications, financials, family content, health, voice, decisions, preferences — these are why animus exists.

### What the employer should know about animus (minimal)

- Public-facing: animus exists, lives at github.com/AreteDriver/animus-docs (private repo with public docs), has a public protocol layer (Quorum / convergentAI on PyPI), is referenced in the operator's portfolio.
- Not disclosed: secrets store contents, integration list, message-gateway adapter list, what runs where, daily usage patterns, accumulated personal memory, the operator's reflection log.

If an employer asks for technical detail on animus during an interview, the answer surface is: architecture diagram + Quorum spec + animus-docs README. That's the public artifact set. Deeper specifics are personal IP and don't get discussed.

---

## Tooling overlap

### If a role provides AI tooling

- Use the work-provided tools for work tasks. They're paid for; they're the right answer for work code.
- Do NOT bridge work tooling into animus (e.g., do not configure animus to read employer Slack via the work-issued Slack token).
- Do NOT bridge animus into work tooling (e.g., do not configure work Claude Enterprise to query memboot indexes on personal hardware).

### If a role requires AI tooling but doesn't provide it

- Build a minimal work-specific assistant on the work-issued hardware using work-provided cloud accounts (or recommend the employer provision them).
- This minimal assistant is work product. It belongs to the employer. Don't transfer animus components into it.
- Code reuse is OK at the *pattern* level (you brought the experience), not at the *artifact* level (don't copy-paste from animus repos into work repos).

### Skills, plugins, MCP servers

- Personal Claude Code skill library (`~/.claude/skills/`, sourced from `ai-skills` public repo) is portable. The skills are public; using them at work is fine.
- Custom skills built specifically for work tasks live in the work environment and stay there.
- Personal MCP servers running on personal hardware (memboot, aurora-query, etc.) are not exposed to work hardware. No tunnels, no proxies, no shared endpoints.

---

## Cross-pollination rules

Some cross-pollination is legitimate and unavoidable. The principle is: **manual, intentional, and one-directional from public-to-private (never the reverse).**

Allowed:
- Lessons learned at work that inform personal-project thinking — written in your own words into personal memory, never pasted from employer source material.
- Public industry information (conference talks, OSS releases, public papers) — fair game in either direction.
- Open-source contributions you make in your free time to projects unrelated to the employer — personal IP, document the time and venue.

Not allowed:
- Pulling employer documents / code / data into personal memory.
- Demonstrating personal projects on employer-issued hardware (creates ambiguity about who built what during what time).
- Using employer cloud accounts for personal experiments.
- Pasting personal portfolio code into employer code reviews.

---

## Offboarding procedure

If an employment relationship ends (voluntary, involuntary, contract end, role pivot), execute:

1. **On work hardware:**
   - Sign out of all personal accounts (Gmail, GitHub personal, personal Slack, Discord, Signal, etc.)
   - Delete any personal data that's accumulated (browser bookmarks, downloads, screenshots)
   - Surrender hardware per employer policy
2. **On personal hardware:**
   - Remove any cached employer SSO sessions
   - Rotate any credentials that touched both sides (rare, but possible — e.g., if personal Anthropic was ever used on work hardware, rotate)
   - Archive any communication with the employer (offer letter, contracts, performance review) to encrypted personal record
3. **In animus:**
   - Search for `work-context` tagged memories — should be empty. If not, archive offline.
   - No animus shutdown needed (it doesn't touch work; it kept running through the role).
4. **Communications:**
   - Update LinkedIn after release date per any non-disclosure agreements
   - Reference protocol per policy: IBM colleagues + project collaborators + OSS contributors. Never employer management.

---

## Communication rules

### In interviews

- Animus exists and is referenceable. Public artifacts: animus-docs README, Quorum spec + convergentAI on PyPI, profile README claims (15K tests, 97% coverage).
- Animus runs on personal hardware. Built on personal time. Not derivative of any prior employer's IP.
- Test counts and coverage numbers are verifiable via the public spec + reference implementation; the proprietary integrator is not browsable.
- If pressed for specifics on private layers: "Forge, Core, and Bootstrap are the proprietary integrator — I can describe the architecture but the source isn't public."

### Inside a current/future role

- Animus is not a topic for work conversation unless an explicit need arises (e.g., the employer asks about prior projects). Default: don't bring it up.
- Don't demo animus on work hardware. Don't screen-share personal tools during work calls.
- If a coworker asks "what tools do you use," answer at the category level ("I have my own assistant infrastructure I use on personal time") not the artifact level.

### Public surfaces (Substack, LinkedIn, Twitter/X, conference talks)

- Discussing animus architecture publicly is fine — it's personal IP.
- Discussing employer specifics requires employer approval per company policy. Default: don't.
- TIAID writing about deployment methodology is fine. Specific anonymized case studies require client consent.

---

## Audit + verification

Quarterly checks to confirm the boundary holds:

1. `grep -r "work-context"` against animus memory — should return empty.
2. Check secrets.env for any employer-named keys — should return empty.
3. Check git config global on personal hardware — should be personal identity.
4. If currently in a role: ssh into work hardware (if remote), verify no personal SSH keys / API keys present.
5. Review recent animus audit log for any tool calls that look work-adjacent.

If any check fails, treat as a boundary breach. Document in `LEARNED.md`, fix immediately, rotate affected credentials, and update this policy if a new failure mode emerged.

---

## Why preemptive

The cost of writing this policy now (low) vs the cost of resolving an ambiguous boundary case after a role lands (high — possibly legal) makes preemption obvious. Setting the rules before they're tested is the only way they stay clear.

Also: writing it down means future-you (or a family member acting on your behalf) has documentation of intent if questions ever arise about IP ownership, separation of personal vs work effort, or the integrity of personal infrastructure during an employment relationship.

---

## What this policy is NOT

- Not legal advice. Consult an employment attorney for jurisdiction-specific IP and confidentiality questions.
- Not a substitute for an employer-specific NDA / employment agreement review. Read those when offered; this policy is the *additional* discipline layered on top of whatever the employer requires.
- Not a refusal to engage with work tooling. Modern jobs use AI tools; this policy says use them appropriately on the work side, not that they're forbidden.
- Not paranoia. It's the same operational discipline a professional applies to any infrastructure where mixing concerns has a cost.

---

## Changelog

- **2026-05-15:** v1.0 — initial policy. Track 9 of PERSONAL_ROADMAP.md. Authored preemptively before any FDE / SE role lands.
