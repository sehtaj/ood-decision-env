# 🤖 AGENT EXECUTION GUIDE (MANDATORY)

This file defines how any AI agent (Codex/GPT) must operate in this repository.

---

# 1. SOURCE OF TRUTH

- The file `rules.md` is the PRIMARY execution contract.
- You MUST read and follow `rules.md` before making any changes.
- If there is any conflict, `rules.md` takes priority.

---

# 2. WORKFLOW (NON-NEGOTIABLE)

For ANY task:

1. Read `rules.md`
2. Understand constraints
3. Make a plan BEFORE coding
4. Implement changes
5. Verify ALL constraints are satisfied

---

# 3. DEFINITION OF DONE

A task is ONLY complete if:

- OpenEnv spec is satisfied
- `openenv validate` passes
- Logging format is EXACT
- Determinism is preserved
- No rules in `rules.md` are violated

---

# 4. FORBIDDEN ACTIONS

You MUST NOT:

- Break logging format
- Introduce randomness
- Change action space (accept/reject/review)
- Ignore validator requirements
- Skip testing or verification

---

# 5. VALIDATION CHECKLIST (MANDATORY)

Before finishing ANY task, ensure:

- Environment runs with reset() and step()
- All 3 tasks work
- Rewards behave correctly
- No crashes occur
- Output matches required format

---

# 6. EXECUTION STYLE

- Think step-by-step before coding
- Prefer minimal, correct changes
- Do NOT over-engineer
- Do NOT modify unrelated files