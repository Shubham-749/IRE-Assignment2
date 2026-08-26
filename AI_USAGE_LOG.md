# AI Usage Log

**Tool:** Claude Code (Anthropic), model Claude Sonnet 5.
**Scope:** Code generation for this assignment was LLM-based throughout, as expected
for this project. Every file under `src/`, `scripts/`, and `tests/`, the PDF/report
generation scripts, and this log itself were written by the AI. What follows is a
chronological account of the actual prompts, direction, and technical decisions that
shaped that code — not a claim that any of it was hand-typed.

The AI's role was implementation: designing modules, writing code, running it against
real downloaded data, debugging failures, and reporting results. The student's role was
direction and judgment: setting scope, making the calls the AI explicitly surfaced,
supplying information only available outside the AI's own tools (platform UIs,
screenshots, account access, real-time system state), and catching gaps the AI missed.
Specific instances of each are called out by name below, not just asserted in general.

A full raw transcript of this session (every prompt and tool call, not just the
technical decisions summarized here) is available via this chat platform's own
session history/export feature if a grader wants the unabridged version — this log is
a curated summary organized for readability, not a substitute for that export.

---

## Q1 — Reproducible Data Pipeline

Student asked for a plain-language summary of the assignment before any code was
written, then asked the AI to scope Q1 as a concrete implementation plan before
touching anything. Two real decisions the student made at that stage:

- **Chose to actually run the pipeline against real downloaded data on this machine**,
  not just write code that type-checks — rejecting the lighter "build code only, run
  it wherever you do heavy compute" option the AI had also offered.
- **Personally authenticated with the gated MIND HuggingFace dataset** (`hf auth
  login`, accepting the dataset's access terms in a browser) — this needed the
  student's own credentials and could not be done by the AI.

AI designed and wrote the unified schema, download/clean/split/feature-store modules,
the one-command CLI, and the leakage tests, then actually executed the pipeline
end-to-end and reported real row counts back for the student to sanity-check against
the exploratory notebooks already on hand. A real bug (MIND's raw impression IDs
colliding across train/val splits) was found during that real run, not anticipated in
advance, and fixed with the student's plan already approved.

## Q2 — BM25 Lexical Retrieval

Student directed the AI to start Q2 directly, having approved the overall project
shape in Q1. AI designed the from-scratch inverted index (a deliberate choice over a
library, matching the assignment's explicit ask), built it, and ran it against real
MIND-small and EB-NeRD-demo data, reporting recall@K numbers back unprompted for
review rather than waiting to be asked.

## Q3 — Semantic (Embedding) Retrieval

Key decision surfaced to the student and made explicitly: **real multilingual
sentence-transformer embeddings vs. a lightweight TF-IDF+SVD alternative**. The AI
laid out the real cost (new dependencies, ~1-2GB, several minutes of compute) against
the lighter option's weaker semantic quality; the student chose the heavier, real
option. AI implemented it, ran it, and reported the resulting recall@K and a genuine
head-to-head comparison against BM25 (which the student had not asked for explicitly
but which the AI judged was implied by "compare lexical vs. semantic" in the
assignment brief).

## Q4 — Offline Evaluation Harness

Student directed the AI to proceed to Q4. AI implemented AUC/MRR/nDCG and the
beyond-accuracy metrics from scratch, verified each against hand-computed toy examples
before trusting them on real data, and ran the harness on both retrievers and both
datasets. No student intervention was needed mid-task here beyond the initial
direction — this was the most self-contained of the six questions.

## Repository and Git Conventions

- Student supplied the actual target GitHub repository URL and asked for the local
  work to be pushed there.
- Student caught that commits were being authored with a `Co-Authored-By: Claude`
  trailer and **explicitly asked for that to stop**, and separately asked for **more
  understandable, less jargon-heavy commit messages** going forward — both were
  applied to every commit from that point on.
- Student later asked for the existing Q1+Q2 commit history to be **split into two
  separate, individually-verified commits** and explicitly authorized the force-push
  this required (a destructive operation the AI would not perform without that
  explicit ask).

## Q5 — Codabench Submissions (the largest share of real debugging)

This question involved the most back-and-forth, almost all of it because real
external systems (Codabench's grader, this machine's actual memory limits) behaved in
ways that could only be discovered by actually running the full-scale pipeline, not by
reasoning about it in advance.

**MIND submission, rejected twice on the real platform:**
- First rejection: the student relayed Codabench's actual error output (a Python
  traceback pasted from the platform) showing the grader expected a file literally
  named `prediction.txt` inside the zip. AI diagnosed and fixed this from the
  student-supplied error text.
- Second rejection: the student again relayed the real platform error (`Inconsistent
  Impression Id 10 and 2`). AI traced this to a lexicographic-sort bug and fixed it,
  adding a regression test so it could not resurface on the EB-NeRD pipeline.
- Student then verified the fix by re-uploading to the real platform and reporting the
  actual leaderboard result each time — nothing here could have been confirmed without
  that manual verification step.

**EB-NeRD's large-scale run, which repeatedly crashed:**
- The AI's own memory-pressure theory was tested and repeatedly failed; what actually
  resolved it was **real-time system information only the student could supply** —
  telling the AI to check current free memory, later reporting that the laptop's
  battery had dropped below 10% partway through a run (a real and correct hypothesis
  for the erratic slowdowns that followed), and proactively closing Chrome/YouTube and
  plugging in the charger between runs at the AI's suggestion.
- Student asked directly, **"are we compromising on result quality by any of the
  workarounds/different attempts you have made so far?"** — this single question is
  what prompted the AI to go back and formally verify (not just assume) that a
  performance fix hadn't silently changed results, which caught a second, more subtle
  correctness gap (a missing dedup step affecting ~1.4% of users) that would otherwise
  have shipped unnoticed.
- Student then chose the specific recovery strategy ("let's first try with 50k
  chunks... if nothing gets compromised that way") after that quality question was
  answered with evidence, not before.
- Student personally checked Codabench's Submission Guidelines page directly (which
  the AI's own web-fetch tools could not render, being a login-gated JS app) and
  reported back the exact required filename, avoiding a repeat of the MIND mistake on
  a much larger, more expensive submission.

**Iterating on MIND's real score:**
- Student directed: **"try to improve score of mind submission. If we get better score
  we'll commit else rollback."** This explicit before/after evidence-gate is what
  shaped the entire approach — the AI validated a proposed model swap offline first,
  specifically to earn the right to spend ~90 minutes of full-scale recompute, rather
  than trying it blind.
- Student confirmed EB-NeRD's own submission was on the real leaderboard as
  "Submitted"/"Running" (again, something only visible from the platform UI) and later
  relayed a course-staff reply (from a TA channel) confirming the platform issue was
  known and being investigated — this directly shaped the decision to stop waiting and
  move on to Q6 rather than keep retrying.
- Student then explicitly asked the AI to also attempt the same kind of improvement
  for EB-NeRD, **while being upfront that they could not verify it themselves this
  time** ("i personally can't verify since platform is having issues... last time your
  theoretical analysis was correct and got reflected on platform too, so maybe
  meanwhile we can do this") — extending trust based on the MIND precedent, not
  assuming it. The resulting offline test was negative, and the AI recommended (and
  the student accepted) not spending the expensive resubmission cycle on it — an
  honest negative result kept in the record rather than discarded.
- Student verified the final submission zip's filename directly against the platform's
  own stated requirement before considering the task closed.

## Q6 — Design Note

- Student chose the scope (design note first, AI usage log after) and format (PDF)
  when the AI presented both as open questions.
- AI drafted the content and a PDF-generation script from the real numbers already
  produced across Q1-Q5.
- Student pointed out the draft was missing the assignment's explicit screenshot
  requirement, then supplied the real screenshot files from their Desktop with enough
  description ("two are from MIND — one showing two different (one improved)
  submissions and other with rank, one for EB-NeRD's submitted status proof") for the
  AI to identify and correctly place each one — including catching, when the images
  first failed to render, that macOS screenshot filenames use a non-standard Unicode
  space character the AI's own hardcoded paths didn't match.
- Student reviewed the rendered PDF and approved it ("design note is very good")
  before it was committed.

## Q9 — Anti-Gaming Ablation

- Student noticed the AI's own earlier work had only satisfied half of Q9's explicit
  two-part requirement (a leakage-boundary test existed, but "report metrics with and
  without features unavailable at serving time" had not been built as a separate
  reported ablation) and said **"yes build that ablation"** — a case of the student
  catching a gap the AI itself had surfaced but not yet closed.
- AI designed the leaked feature (a user's clicks from their other validation-split
  impressions, added to their query) and ran it, reproducing Q4's published metrics
  exactly in the "without leak" condition as a built-in correctness check before
  trusting the comparison.
- When the AI then asked the student a procedural question (whether to also commit a
  results CSV), the student redirected: **"i don't understand this, do as per
  assignment requirements"** — rather than explain the tradeoff, the AI went back to
  the actual assignment PDF (`A1.pdf`, on disk in the repo) and confirmed Q9's exact
  wording, which settled the question directly instead of asking the student to choose
  between options they had no way to evaluate. This is a genuinely different pattern
  from earlier in the project, where the student made calls the AI explicitly
  presented (PDF vs. other format, etc.) — here the right move was for the AI to
  resolve it itself from the primary source, and the student's push was what caused
  that correction.
- Real result reported honestly either way: the leaked feature mostly makes metrics
  slightly *worse*, not better, across 3 of 4 dataset/retriever combinations — only
  EB-NeRD embeddings showed a small, consistent gain. Added to both the design note
  (§7) and `results/leakage_ablation.csv` without softening or cherry-picking the
  negative result.

## Code Provenance Summary

Every `.py` file under `src/ire_a1/`, `scripts/`, and `tests/`, the PDF-generation
scripts, `README.md`'s technical content, and this log were written by the AI. No
hand-written source code exists in this repository. The student's contribution is the
direction, decisions, and verification described above — most concretely visible in
this repository's git history as a sequence of AI-authored commits, each triggered by
an explicit student instruction, and in the real, externally-verified results (actual
Codabench scores, actual platform error messages, actual screenshots) that could only
have entered this project through the student's own actions outside the AI's tools.
