# How to Validate Security Findings
## API Attack Path Analyzer — Field Reference

Use this file while working through findings in the HTML report.
Each section maps directly to what you see on screen.

---

## The Core Question for Every Finding

> "Does the spec say this endpoint is public/unprotected, and does the server actually behave that way?"

If both are true → confirmed vulnerability.
If spec says public but server enforces auth → spec gap (still worth fixing).
If spec says protected but tool flagged it → false positive, check the narrative.

---

## Step 1 — Find the Endpoint in the Spec

The report shows each step as `METHOD:/path` (e.g. `GET:/store/order/{orderId}`).
Open your spec file and search for that path and method.

```yaml
paths:
  /store/order/{orderId}:    # search for this
    get:                     # then this method
      summary: Find purchase order by ID
      security: []           # THIS is what you are checking
```

---

## Step 2 — Verify the Auth Claim

| What you see in the spec | Meaning | What to do |
|---|---|---|
| No `security:` key at all | Treated as public by the tool | Test without a token |
| `security: []` (empty list) | Explicitly public | Test without a token |
| `security: [{bearerAuth: []}]` | Auth required per spec | Tool should not flag as unauthenticated — re-read the narrative |
| Global `security:` at root level | Applies to all operations unless overridden | Check if the specific operation overrides it |

Look for the global security at the top of the spec:
```yaml
security:           # global — applies to every operation
  - bearerAuth: []

paths:
  /public/health:
    get:
      security: []  # this operation overrides global → public
```

---

## Step 3 — Check What Data the Endpoint Returns

Find the response schema and follow any `$ref`:

```yaml
responses:
  '200':
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/Order'   # follow this
```

Then find `Order` under `components/schemas` and read its fields.

Ask these questions:
- Does this return data that belongs to a specific user/account?
- Does it include PII (email, name, address, phone, SSN)?
- Does it include credentials, tokens, or internal IDs?
- Could an attacker use this output as input to the next step?

If yes to any → the data exposure is meaningful and the finding is worth testing.

---

## Step 4 — Test With curl (No Auth First)

Always test the **first step of the chain without any token**.
If it returns data without auth, that alone may be the vulnerability.

```bash
# Basic unauthenticated test
curl -v https://your-api.example.com/store/order/1

# BOLA test — increment the ID
curl https://your-api.example.com/store/order/1
curl https://your-api.example.com/store/order/2
curl https://your-api.example.com/store/order/3

# If you need a token for a later step, get one first
curl -X POST https://your-api.example.com/user/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "test123"}'
# Copy the token from the response, then use it:
curl -H "Authorization: Bearer <token>" \
  https://your-api.example.com/admin/users/999

# Test a write operation (mass assignment)
curl -X PUT https://your-api.example.com/users/me \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"username": "me", "role": "admin", "isAdmin": true}'
```

---

## Step 5 — Walk the Full Chain in Order

The report lists steps 1, 2, 3… in order.
Each step has an **Attacker gains** field — that output becomes the input for the next step.

Example chain walk:

```
Step 1: GET /users?page=1  (no auth)
  → Attacker gains: list of user IDs [101, 102, 103, ...]

Step 2: GET /users/102/profile  (no auth, BOLA)
  → Attacker gains: email, phone, address of user 102

Step 3: POST /auth/password-reset  (no auth, no rate limit)
  → Input: email from step 2
  → Attacker gains: account takeover for user 102
```

Follow this literally — call each endpoint in order and confirm the output
of each step is actually achievable before moving to the next.

---

## Step 6 — Check the Path Parameters

Integer path parameters are the primary BOLA/IDOR signal.
Check the parameter type in the spec:

```yaml
parameters:
  - name: petId
    in: path
    required: true
    schema:
      type: integer      # HIGH risk — trivially enumerable
      format: int64

  - name: userId
    in: path
    schema:
      type: string
      format: uuid       # lower risk — but not zero (some DBs use sequential UUIDs)
```

If `type: integer` and the endpoint returns user-specific data with no auth → very likely BOLA.

---

## Step 7 — Record Your Verdict

Keep a log entry for each finding you test:

```
Finding name:     [copy from report]
Severity:         CRITICAL / HIGH / MEDIUM / LOW
OWASP category:   API1:2023 / API2:2023 / etc.

Spec check:
  - Endpoint:       GET /store/order/{orderId}
  - security: key:  missing (public)
  - Path param:     orderId — type: integer
  - Response data:  Order object with petId, quantity, userId — user-specific

curl test:
  - Command:  curl https://api/store/order/2
  - Result:   200 OK, returned order belonging to another user
  - Reproduced: YES

Chain reproduced end-to-end: YES / NO / PARTIAL

Verdict:
  [ ] TRUE POSITIVE   — server behaves as spec claims, attack chain reproduced
  [ ] SPEC GAP        — server actually enforces auth, but spec doesn't document it
  [ ] FALSE POSITIVE  — chain requires context attacker can't realistically have
  [ ] NEEDS MORE INFO — can't test without access to a running server

Remediation:
  1. Add security: [{api_key: []}] to GET /store/order/{orderId}
  2. Add server-side ownership check: order.userId must equal authenticated user ID
```

---

## Quick Reference — What to Check per OWASP Category

### API1:2023 — BOLA / IDOR
- Integer or UUID path parameter?
- No ownership check in spec or server?
- Can you access resource IDs belonging to other users?

```bash
curl /api/orders/1   # your order
curl /api/orders/2   # someone else's order — if 200, confirmed
```

### API2:2023 — Broken Authentication
- Missing `security:` on a sensitive operation?
- Weak auth scheme (API key in query param instead of header)?
- Password in query string?

```bash
curl /api/user/login?username=test&password=test   # credentials in URL = logged everywhere
curl /api/admin/users                               # no token — if 200, confirmed
```

### API3:2023 — Broken Object Property Level Authorization
- Does the request body schema include `role`, `isAdmin`, `permissions`, `status`?
- Can a regular user write to those fields?

```bash
curl -X PATCH /api/users/me \
  -d '{"role": "admin"}'   # if server accepts it, confirmed
```

### API4:2023 — Unrestricted Resource Consumption
- No rate limit headers in spec?
- Expensive operation (bulk export, password reset, OTP send)?

```bash
# Send 20 requests in quick succession, check if any are rejected
for i in $(seq 1 20); do curl /api/auth/reset -d '{"email":"victim@test.com"}'; done
```

### API5:2023 — Broken Function Level Authorization
- Admin-looking path (`/admin/`, `/internal/`, `/management/`) with no auth?
- Different method on same path with different security?

```bash
curl /api/admin/users           # regular user token — if 200, confirmed
curl -X DELETE /api/admin/users/5
```

### API6:2023 — Unrestricted Access to Sensitive Business Flows
- Can you bypass quantity limits, referral caps, discount codes?
- No validation described in request body schema?

```bash
curl -X POST /api/orders -d '{"quantity": 999999, "price": 0.01}'
```

### API8:2023 — Security Misconfiguration / SSRF
- Parameter named `url`, `callback`, `endpoint`, `redirect`, `webhook`, `target`?
- Does the server make outbound requests based on user input?

```bash
curl /api/fetch?url=http://169.254.169.254/latest/meta-data/   # AWS metadata
curl /api/webhook -d '{"url": "http://internal-service.local/"}'
```

---

## Severity and What to Do With It

| Severity | Typical finding | Reproduce within | Report to |
|---|---|---|---|
| CRITICAL | Account takeover, mass data export, auth bypass | Same day | Engineering lead immediately |
| HIGH | BOLA on user data, privilege escalation | This week | Security ticket, current sprint |
| MEDIUM | Excessive data exposure, missing rate limits | This month | Backlog, next security review |
| LOW | Credentials in query params, verbose errors | Next quarter | Improvement ticket |

---

## Common False Positive Signals

These suggest the finding is not exploitable in practice — still worth noting:

- **Server returns 401/403 despite spec showing no `security:`**
  → Auth is enforced in middleware not reflected in the spec. Update the spec.

- **Endpoint is on an internal network path only**
  → Not reachable from the internet. Still a risk if an attacker gets internal access.

- **Chain requires the attacker to already have a valid account**
  → Check if account registration is open or invite-only. Open registration lowers the bar significantly.

- **Response returns data but it is all public/non-sensitive**
  → A public product catalog being accessible without auth is not a finding.
  → Check the `sensitivity_class` in the report — if it says PUBLIC, the tool may have over-ranked it.

- **The `llm_self_score` in the confidence breakdown is below 0.5**
  → Claude was not confident. Treat as a lead, not a confirmed finding.

---

## Useful Commands Reference

```bash
# Check what auth schemes are declared in the spec
grep -A5 "securitySchemes" spec.yaml

# Find all operations with no security declaration
grep -B10 "operationId" spec.yaml | grep -v "security"

# Find all integer path parameters
grep -A3 "in: path" spec.yaml | grep "type: integer"

# Check response headers for rate limiting
curl -I https://your-api.example.com/auth/login

# Decode a JWT token to inspect claims (no verification)
echo "<token>" | cut -d. -f2 | base64 -d 2>/dev/null | python3 -m json.tool
```

---

*Generated by API Attack Path Analyzer — keep this file next to your spec during security reviews.*
