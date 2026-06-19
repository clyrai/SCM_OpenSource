# SCM Cloud — API authentication

How accounts and API keys work in the hosted SCM. Self-hosted users can ignore this entirely (set `SCM_CLOUD_AUTH=0` and the open-source unauthenticated shape is preserved).

---

## Token shape

```
Authorization: Bearer scm_live_<keyid>_<secret>
```

Example: `Authorization: Bearer scm_live_a1b2c3d4e5f6_8z9y8x7w6v5u4t3s2r1q0p9o8n7m6l5k`

- **Prefix `scm_live_`** so anyone scanning logs can spot a leaked SCM key on sight.
- **`<keyid>`** (12 chars, lowercase base32) is logged for audit and shown in the dashboard so you can identify a key without holding the full secret.
- **`<secret>`** (32 chars, lowercase base32) is the actual bearer token. **Returned exactly once** at issuance, never echoed back from any read endpoint. Lose it → revoke and mint a new one.

The full token is hashed (`sha256`) before storage. Losing the database does not expose live keys.

---

## Bootstrap a new account

### 1. Create the account

```bash
curl -X POST https://scm.run/v1/cloud/accounts \
     -H "Content-Type: application/json" \
     -d '{"email":"alex@example.com"}'
# {"id":"acct_a1b2c3d4...", "email":"alex@example.com", "tier":"free", ...}
```

If the deployment has `SCM_CLOUD_SIGNUP_TOKEN` set (recommended for any public deploy), include it:

```bash
curl -X POST https://scm.run/v1/cloud/accounts \
     -H "X-Signup-Token: $SIGNUP_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"email":"alex@example.com"}'
```

### 2. Mint the first API key (no auth required, but only once per account)

```bash
curl -X POST https://scm.run/v1/cloud/accounts/$ACCOUNT_ID/keys/initial \
     -H "Content-Type: application/json" \
     -d '{"label":"laptop"}'
# {
#   "id":"key_xyz...",
#   "key_prefix":"scm_live_a1b2c3d4e5f6",
#   "label":"laptop",
#   "scopes":"memories:rw",
#   "rate_limit_per_min":60,
#   "created_at":"2026-05-05T...",
#   "token":"scm_live_a1b2c3d4e5f6_8z9y8x7w6v5u4t3s2r1q0p9o8n7m6l5k"  ← save this
# }
```

After this, the `keys/initial` endpoint refuses additional calls (409). Mint further keys via the authed endpoint below.

### 3. Use the key

```bash
curl -X POST https://scm.run/v1/memories \
     -H "Authorization: Bearer scm_live_a1b2c3d4e5f6_8z9y..." \
     -H "Content-Type: application/json" \
     -d '{"text":"My favorite coffee is filter coffee."}'
```

---

## Account-scoped endpoints (require Bearer auth)

```
GET    /v1/cloud/me                 whoami: id, email, tier
GET    /v1/cloud/me/keys            list this account's API keys (no secrets)
POST   /v1/cloud/me/keys            mint another key for this account
DELETE /v1/cloud/me/keys/{key_id}   revoke one key
POST   /v1/cloud/me/byok            set the BYOK LLM provider+key+model
DELETE /v1/cloud/me/byok            clear the BYOK config
```

Mint another key:

```bash
curl -X POST https://scm.run/v1/cloud/me/keys \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"label":"ci-server","rate_limit_per_min":120}'
```

Revoke:

```bash
curl -X DELETE https://scm.run/v1/cloud/me/keys/key_xyz \
     -H "Authorization: Bearer $TOKEN"
```

---

## BYOK LLM configuration

SCM's optional LLM-extraction step (used during sleep cycle to refine concepts) needs an LLM. You bring your own key:

```bash
curl -X POST https://scm.run/v1/cloud/me/byok \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "provider":"deepseek",
       "api_key":"sk-deepseek-key-xxx",
       "base_url":"https://api.deepseek.com",
       "model":"deepseek-chat"
     }'
```

- The `api_key` is **encrypted at rest** under the deployment's `SCM_CLOUD_SECRET_KEY`.
- It is **never returned** in any read endpoint — `GET /v1/cloud/me` does not include it.
- Without a BYOK config set, SCM falls back to its **heuristic extractor** (free, no LLM cost, slightly lower extraction quality). Most users won't notice the difference until they have hundreds of memories.

Clear it:

```bash
curl -X DELETE https://scm.run/v1/cloud/me/byok -H "Authorization: Bearer $TOKEN"
```

---

## Tenant isolation (the safety property that matters)

When cloud auth is on, every `/v1/memories/*` request is automatically namespaced under the calling account. Specifically: the `user_id` parameter you pass becomes `<account_id>::<your_user_id>` server-side, before any SQLite write or read.

What this means in practice:

- **Account A and Account B both use `user_id="default"`.** Their memories are completely isolated. A's facts never appear in B's search results.
- **Account A cannot read account B's data even if they guess B's user_id.** The guess is ignored — A's request is namespaced under A.
- **A leaked API key only leaks data for that one account.** Other accounts are untouched.

This is enforced by middleware ([`src/cloud/auth_middleware.py`](../src/cloud/auth_middleware.py)) before the handler runs, and verified by `test_cross_tenant_isolation` in [`tests/test_cloud_auth.py`](../tests/test_cloud_auth.py).

---

## Rate limits

Each API key has a `rate_limit_per_min` (default 60, configurable per key on issuance).

- Implemented as an **in-process token bucket** ([`src/cloud/rate_limit.py`](../src/cloud/rate_limit.py)).
- Smooth across small bursts (you can hit the API several times in a row at the start of a turn).
- Sustained throughput is bounded.
- Exhausting the bucket returns **HTTP 429** with `{"error":"rate limit exceeded","limit_per_min":N}`.

For multi-machine deployments, swap the in-process bucket for a Redis-backed one (the `acquire(key_id, capacity)` interface is unchanged).

---

## Public (unauthenticated) routes

These work regardless of `SCM_CLOUD_AUTH`:

| Path | Why |
|---|---|
| `GET /v1/health` | Liveness probe, status-page integration |
| `GET /v1/tools` | Tool-definition export (OpenAI/Anthropic/Gemini formats) |
| `GET /v1/openapi.json` | OpenAPI 3.1 spec for ChatGPT Custom GPT Actions |
| `POST /v1/cloud/accounts` | Signup (gated by `SCM_CLOUD_SIGNUP_TOKEN` if set) |
| `POST /v1/cloud/accounts/{id}/keys/initial` | First-key bootstrap |
| `GET /demo/...` | The public demo UI |

---

## Self-hosted (no-auth) mode

For users running their own SCM server who don't need multi-tenancy:

```bash
# Don't set SCM_CLOUD_AUTH (or set it to 0)
scm serve --port 8000
```

In this mode `/v1/memories/*` accepts requests with no Authorization header. The `user_id` parameter is honored as-is (no namespacing). This is the open-source shape from before v0.7.8.

---

## Operational checklist before exposing SCM Cloud publicly

- [ ] `SCM_CLOUD_AUTH=1` in the deploy environment
- [ ] `SCM_CLOUD_SECRET_KEY` set to a real 32+ byte secret (used to encrypt BYOK keys at rest)
- [ ] `SCM_CLOUD_SIGNUP_TOKEN` set to gate signups (otherwise anyone can spam `/v1/cloud/accounts`)
- [ ] HTTPS only (Fly / Railway / your CDN handles this)
- [ ] Postgres for production storage (SQLite is fine for one server, becomes a bottleneck past ~100 active users)
- [ ] Rate limit defaults reviewed — the per-key default of 60/min may need to be lower for free-tier abuse mitigation
- [ ] Monitoring on `/v1/health` (UptimeRobot / Better Uptime)
- [ ] Logs forwarded to a real sink (Sentry / Datadog / CloudWatch)
- [ ] A documented rotation procedure for `SCM_CLOUD_SECRET_KEY` (rotating invalidates existing BYOK encrypted keys; users re-set them)

---

## What this enables

With v0.7.8 (Phase 1 of SCM Cloud) shipped, the things that previously needed engineer-glue now have proper plumbing:

- **Multi-tenant SaaS** — each customer is an `account_id`, fully isolated, with their own keys and BYOK config
- **Audit trail** — every key has `created_at` / `last_used_at` so you can see which key did what
- **Key rotation** — mint a new key, revoke the old one, no downtime
- **Free / paid tiers** — the `tier` column on `accounts` is the hook; Phase 3 adds Stripe enforcement
- **BYOK** — users own the LLM cost; the cloud charges only for memory infrastructure
