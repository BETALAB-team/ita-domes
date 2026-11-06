# Claim author task packet

Requested model: **gpt-6-sol**, fresh context. Read the full human request at `00_context/user_request.md` and the selected scope in `01_plan/research_questions.md` and `decisions.md`.

Own `06_claim_drafts` and author responses plus issue-register files in `07_reviews`. Do not edit reviewers' files. Keep every draft immutable. Initial draft is `article_claims_v00.md`; four revisions are v01–v04. Further revisions are permitted for real unresolved substantive issues.

Use the requested atomic schema for every stable claim ID, covering literature; gap/novelty; research questions; data; methods; results; interpretation; limitations. Roughly 20–35 well-bounded claims are preferable to redundant fragmentation, but completeness governs. No polished article paragraphs. Put source IDs, exact locators or input/script/run/output paths in every supported claim. Preserve published versus newly computed results. Use explicit planned/unverified claims for work still pending.

Read all completed discovery evidence, source ledgers, methods protocols, run metadata/checks and verified tables. Critically verify relevant evidence yourself. Existing research drafts are leads, not evidence. The primary article is the controlled ambiguity of survey-block-hour disaggregation, not temporal predictive transfer, clustering, retrofit savings, or a claimed correction to CLIMA.

Create machine-readable `claim_ledger_v00.json` (and one per revision) with matching IDs, exact claims, kind, evidence class/status, source IDs, locators, outputs, dependencies and permitted wording. Maintain `claim_evidence_ledger.json` as latest indexed snapshot with draft filename and SHA256. Link every central result to checked outputs. Missing measurements do not block a truthful incomplete research draft, but empirical validation may not be asserted.

For each review round N, independently inspect its findings/evidence. Save `07_reviews/response_NN.md` with issue-by-issue change location/evidence, reasoned rebuttal, or explicit unresolved dependency. Save `07_reviews/issue_register_NN.json` and update readable `issue_register.md` plus latest `issue_register.json`. Each entry names source round, issue ID, severity, claim IDs, disposition, revision/evidence, and whether still open. Earlier valid limitations remain visible.

Every cycle must produce a new substantive revision responding to review, not cosmetic version inflation. If a reviewer finds no errors, clarify the tested scope/evidence where scientifically useful and document no invented criticism. Record vNN hash and prior input hash. Report concrete changes, open dependencies, and output paths to orchestrator. Do not write prose or final accepted snapshot until closure/gate.
