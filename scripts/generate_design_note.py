#!/usr/bin/env python
"""Q6: generate the design note PDF from real numbers already produced by Q1-Q5.

    python scripts/generate_design_note.py

Writes design_note.pdf at the repo root. Content mirrors README.md's Design notes
sections but condensed to fit the assignment's <=4 page limit.
"""

import glob
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader

# Real Codabench screenshots, embedded in section 5.
def _find(glob_pattern: str) -> str | None:
    # macOS screenshot filenames use a narrow no-break space (U+202F) before AM/PM,
    # not a regular space -- glob instead of hardcoding the path so this doesn't
    # silently fail to match (it did, the first time this was written literally).
    matches = glob.glob(glob_pattern)
    if not matches:
        print(f"[warn] no screenshot matched: {glob_pattern}")
        return None
    return matches[0]


MIND_COMPARE_SCREENSHOT = _find("/Users/shubhampaliwal/Desktop/Screenshot 2026-08-25 at 2.46.55*PM.png")
MIND_LEADERBOARD_SCREENSHOT = _find("/Users/shubhampaliwal/Desktop/Screenshot 2026-08-25 at 2.47.13*PM.png")
EBNERD_STATUS_SCREENSHOT = _find("/Users/shubhampaliwal/Desktop/Screenshot 2026-08-26 at 1.45.52*PM.png")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "design_note.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=11.3, spaceAfter=4)
tight = ParagraphStyle("tight", parent=body, spaceAfter=2)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13.5, leading=16, spaceBefore=0, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=3,
                     textColor=colors.HexColor("#1a1a1a"))
title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=15, leading=18, spaceAfter=2)
subtitle_style = ParagraphStyle("subtitle", parent=body, fontSize=9, textColor=colors.HexColor("#555555"),
                                 spaceAfter=10)
caption = ParagraphStyle("caption", parent=body, fontSize=7.8, leading=9.5, textColor=colors.HexColor("#444444"),
                          spaceAfter=8)


def bullets(items, style=tight):
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=0) for t in items],
        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=7, spaceAfter=4,
    )


def table(rows, col_widths, font_size=7.6, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def screenshot(path, max_width=3.4 * inch, cap=None):
    if not path or not Path(path).exists():
        return None
    reader = ImageReader(path)
    iw, ih = reader.getSize()
    w = min(max_width, iw)
    h = ih * (w / iw)
    flow = [RLImage(path, width=w, height=h)]
    if cap:
        flow.append(Paragraph(cap, caption))
    return KeepTogether(flow)


story = []

story.append(Paragraph("Lexical &amp; Semantic Retrieval on MIND and EB-NeRD", title_style))
story.append(Paragraph(
    "CS4.406 Information Retrieval &amp; Extraction — Assignment 1 Design Note &nbsp;|&nbsp; "
    "Shubham Paliwal &nbsp;|&nbsp; github.com/Shubham-749/IRE-Assignment1",
    subtitle_style))

# ---------------------------------------------------------------- 1. Overview
story.append(Paragraph("1. Overview", h1))
story.append(Paragraph(
    "Built a reproducible pipeline for both MIND (English) and EB-NeRD (Danish) covering: a unified "
    "data pipeline with temporal (never random) splitting and a leakage-safe point-in-time history "
    "accessor (Q1); lexical (BM25) and semantic (embedding) candidate generation (Q2/Q3); an offline "
    "evaluation harness with the official ranking metrics, beyond-accuracy metrics, and bootstrap "
    "confidence intervals (Q4); and real Codabench submissions, including one real, evidence-based "
    "improvement iteration on MIND (Q5). Every number below is measured against the real downloaded "
    "datasets and real leaderboard scores, not simulated.", body))

# ---------------------------------------------------------- 2. Pipeline & design choices
story.append(Paragraph("2. Pipeline and Design Choices (Q1&ndash;Q3)", h1))

story.append(Paragraph("Data pipeline &amp; leakage boundary.", h2))
story.append(Paragraph(
    "Both datasets already ship official time-ordered train/val folders. Rather than re-split randomly "
    "&mdash; which would leak future clicks into training, the opposite of how a deployed recommender "
    "sees data &mdash; the pipeline validates that the official folders are genuinely temporally "
    "ordered (<code>verify_split_integrity</code>) and reuses them directly. User history lives in one "
    "long-format table, and every retrieval method reads it through a single point-in-time accessor "
    "(<code>get_user_history</code> / <code>UserHistoryIndex</code>) that always filters to clicks "
    "strictly before the impression being scored &mdash; one testable leakage boundary (Q9) instead of "
    "logic duplicated into every feature.", body))

story.append(Paragraph("Lexical retrieval (BM25).", h2))
story.append(Paragraph(
    "A dict-based inverted index and classic Okapi BM25 (k1=1.5, b=0.75) over title+abstract, built "
    "from scratch rather than a library &mdash; simple and transparent at this scale (12K&ndash;125K "
    "articles), and it let the same scoring formula be reused unchanged for both full-corpus retrieval "
    "(Q2) and impression-level candidate scoring (Q4).", body))

story.append(Paragraph("Semantic retrieval (embeddings).", h2))
story.append(Paragraph(
    "A multilingual sentence-transformer (<code>paraphrase-multilingual-MiniLM-L12-v2</code>) so one "
    "model covers both English MIND and Danish EB-NeRD, with brute-force cosine similarity "
    "(L2-normalized matrix + matvec) rather than FAISS &mdash; simpler at 12K&ndash;125K articles and "
    "explicitly allowed by the assignment; FAISS is the natural next step at 10x+ scale (&sect;5). A "
    "user's query vector is the mean-pooled, re-normalized embedding of their recent click history. "
    "<b>Alternative rejected:</b> lightweight TF-IDF+SVD embeddings (no heavy dependency) &mdash; "
    "rejected because it is still built on the same bag-of-words counts as BM25 and unlikely to show "
    "genuine semantic generalization the way a real contextual model can.", body))

# ---------------------------------------------------------- 3. Evaluation harness
story.append(Paragraph("3. Evaluation Harness (Q4)", h1))
story.append(Paragraph(
    "AUC (tie-aware Mann-Whitney rank formula), MRR, and nDCG@{5,10} implemented from scratch, plus "
    "diversity / novelty / coverage, all with bootstrap 95% confidence intervals, and a cold-start "
    "(&lt;5 history items) vs. warm slice. This operates at the <i>impression</i> level &mdash; ranking "
    "the small candidate list Codabench itself scores &mdash; a genuinely different task from Q2/Q3's "
    "full-corpus candidate-generation recall, and both are reported because they can disagree (&sect;4).",
    body))

# ---------------------------------------------------------- 4. Key findings
story.append(Paragraph("4. Key Findings: Lexical vs. Semantic, and Where They Disagree", h1))

story.append(table(
    [["Dataset", "Method", "recall@50", "recall@100", "recall@200", "AUC (Q4)", "Diversity@5"],
     ["MIND-small", "BM25", "0.53%", "1.15%", "2.19%", "0.553", "0.917"],
     ["MIND-small", "Embeddings", "0.88%", "1.43%", "2.27%", "0.633", "0.847"],
     ["EB-NeRD demo", "BM25", "0.83%", "1.55%", "2.76%", "0.494", "0.842"],
     ["EB-NeRD demo", "Embeddings", "0.56%", "1.12%", "2.78%", "0.539", "0.780"]],
    [0.85 * inch, 0.78 * inch, 0.68 * inch, 0.72 * inch, 0.72 * inch, 0.62 * inch, 0.68 * inch],
))
story.append(Spacer(1, 5))

story.append(Paragraph(
    "The two datasets disagree, and that disagreement is the finding, not noise. <b>On MIND, embeddings "
    "win decisively</b> at every K and every ranking metric &mdash; BM25's impression-level AUC (0.553) "
    "is barely above the 0.5 chance level. <b>On EB-NeRD, BM25 keeps a real recall edge</b> at tight "
    "budgets, and at the impression-ranking level embeddings' AUC (0.494 offline) is itself "
    "indistinguishable from chance. Spot-checking examples explains why: BM25 finds articles that "
    "lexically echo a user's history, but real clicks are often on something topically adjacent with "
    "zero word overlap &mdash; the classic lexical-retrieval gap embeddings exist to close. On EB-NeRD, "
    "Danish headlines are terse and formulaic enough that exact keyword match stays a strong signal on "
    "its own, so semantic expressiveness does not pay off the same way. Accuracy and diversity trade off "
    "directly on both datasets too: the more-accurate embedding ranking is consistently the "
    "less-diverse one (tighter thematic clustering) &mdash; a beyond-accuracy metric surfacing a "
    "tradeoff AUC alone never would.", body))

# ---------------------------------------------------------- 5. Codabench + iteration
story.append(Paragraph("5. Codabench Submissions and Iterating on Real Results (Q5)", h1))
story.append(Paragraph(
    "BM25 was not used for either real submission, and this was measured, not assumed: timed at "
    "11&ndash;170ms/impression, it would take on the order of <b>days</b> across MIND's 2.37M or "
    "EB-NeRD's 13.5M impressions. Embedding similarity vectorizes (batched matrix ops) and is the only "
    "method actually tractable at real scale.", body))

story.append(Paragraph("MIND: a real, evidence-based improvement.", h2))
story.append(Paragraph(
    "First submission (AUC 0.6194, MRR 0.3006, nDCG@5 0.3225, nDCG@10 0.3784) used the shared "
    "multilingual embedding model &mdash; a real compromise, since MIND is 100% English and a "
    "multilingual model trades per-language quality for cross-lingual coverage MIND never needed. "
    "Hypothesis: an English-specific model (<code>all-mpnet-base-v2</code>) would help. Validated "
    "cheaply offline on MIND-small <i>before</i> committing to the expensive full-catalog re-embed: AUC "
    "0.6513 vs. 0.6333, 95% confidence intervals non-overlapping ([0.643,0.659] vs. [0.625,0.642]) "
    "&mdash; a real signal, not sampling noise. Re-submitted and confirmed on the live leaderboard: "
    "<b>AUC 0.6194 &rarr; 0.6544</b>, MRR 0.3006 &rarr; 0.3211, nDCG@5 0.3225 &rarr; 0.3464, nDCG@10 "
    "0.3784 &rarr; 0.4023 &mdash; every official metric improved, not just the one the hypothesis "
    "targeted.", body))

mind_compare_shot = screenshot(
    MIND_COMPARE_SCREENSHOT, max_width=4.6 * inch,
    cap="Fig. 1 &mdash; Both MIND submissions: 0.6194 (original) and 0.6544 (after the model swap).",
)
mind_leaderboard_shot = screenshot(
    MIND_LEADERBOARD_SCREENSHOT, max_width=4.6 * inch,
    cap="Fig. 2 &mdash; Improved submission's full leaderboard row (rank shown for reference only; "
        "grading is never on leaderboard rank).",
)
for shot in (mind_compare_shot, mind_leaderboard_shot):
    if shot:
        story.append(Spacer(1, 3))
        story.append(shot)
story.append(Spacer(1, 3))

story.append(Paragraph("EB-NeRD: the same discipline, an honest negative result.", h2))
story.append(Paragraph(
    "The identical lever does not transfer. Since Danish genuinely needs multilingual support, the "
    "analogous move was upgrading model <i>capacity</i> within the multilingual family "
    "(MiniLM&rarr;mpnet-base). Tested offline first, exactly as for MIND &mdash; and the result was "
    "flat-to-negative (AUC 0.539&rarr;0.529, MRR and nDCG also down), consistent with &sect;4's finding "
    "that EB-NeRD's task is more lexical than semantic: a larger embedding model does not fix a "
    "bottleneck it was never the cause of. The expensive full-scale EB-NeRD resubmission was correctly "
    "<i>not</i> attempted on the strength of this result. Reporting a negative finding honestly, and "
    "not spending compute chasing it further, is itself the point of validating offline first.", body))

story.append(Paragraph("EB-NeRD submission status.", h2))
story.append(Paragraph(
    "13,536,710 predictions generated and submitted, fully validated pre-upload (exact expected count, "
    "correct sequential order matching the source file, correct required filename inside the zip). "
    "Codabench's own scoring stage has been stuck on “Running” since submission; course staff "
    "confirmed this is a known platform-side issue currently under investigation with the organizers, "
    "not a problem with the submission itself, and advised keeping the zip file safe for when scoring "
    "resumes. The result will be reported if/when the platform recovers.", body))

eb_shot = screenshot(
    EBNERD_STATUS_SCREENSHOT, max_width=3.8 * inch,
    cap="Fig. 3 &mdash; EB-NeRD submissions on Codabench: accepted (“Submitted”), no score yet "
        "&mdash; the platform-side scoring stall described above.",
)
if eb_shot:
    story.append(Spacer(1, 3))
    story.append(eb_shot)
    story.append(Spacer(1, 3))

# ---------------------------------------------------------- 5b. Testing & verification
story.append(Paragraph("Testing and format verification.", h2))
story.append(Paragraph(
    "48 tests cover BM25/embedding correctness, metric formulas against hand-computed examples, "
    "leakage boundaries, and download idempotency. Two of them exist because a real submission failed "
    "twice, not because we anticipated the bugs: Codabench rejected the first MIND upload for the zip "
    "containing the wrong internal filename (<code>mind_prediction.txt</code> instead of the platform's "
    "hardcoded expected <code>prediction.txt</code>), then rejected the fix for scrambled row order "
    "&mdash; sorting by a string <code>impression_id</code> column put <code>\"10\"</code> right after "
    "<code>\"1\"</code> lexicographically, but Codabench compares predictions to ground truth "
    "<i>positionally</i>, line by line, in the original file's order. Both fixes were centralized into "
    "one shared, tested function (<code>rank_and_group</code>) used by <i>both</i> datasets' submission "
    "scripts, so EB-NeRD's pipeline never had to rediscover either bug independently.", body))

# ---------------------------------------------------------- 6. Where it breaks
story.append(Paragraph("6. Where This Pipeline Breaks at 10x Scale", h1))
story.append(bullets([
    "<b>BM25 does not scale without re-architecting.</b> Measured, not hypothetical: at "
    "11&ndash;170ms/impression it would take days at real Codabench volumes. A batched/vectorized "
    "sparse-matrix BM25 or a real search engine (Lucene/tantivy) would be needed at 10x+.",
    "<b>Brute-force embedding search needs an ANN index at 10x+.</b> Fine at 12K&ndash;125K articles, "
    "but matvec cost grows linearly with catalog size and is already the majority of per-query cost "
    "at 125K articles &mdash; FAISS is the natural next step.",
    "<b>Memory, not just CPU, is the real ceiling at 10x, and it showed up for real.</b> Generating "
    "EB-NeRD's actual submission (13.5M impressions, 807,677 users) repeatedly crashed a 17.2GB "
    "development machine. Root cause: a helper built for leakage-safe point-in-time filtering was "
    "reused for a task that never needed that filtering, silently exploding 807K users' history into "
    "116.8M rows just to deduplicate and immediately re-aggregate it back down. Fixed by reading raw "
    "per-user history directly. The most instructive lesson of the assignment: code correct and "
    "well-tested at small scale can still hide a real cost that only appears once production-scale "
    "data arrives &mdash; and initial profiling suspicion (the scoring loop) was itself wrong, even "
    "with real data and real crashes in front of us.",
    "<b>Single-machine, single-process throughout.</b> No distributed compute anywhere in this "
    "pipeline. Real production scale (all users, daily retraining) would need horizontal scaling "
    "(Spark/Ray for the data pipeline, a served ANN index rather than an in-process matrix) that "
    "nothing here currently provides.",
]))

# ---------------------------------------------------------- 7. Anti-gaming
story.append(Paragraph("7. Anti-Gaming and Leakage (Q9)", h1))
story.append(Paragraph(
    "Every user-history read goes through the one tested point-in-time accessor described in &sect;2; "
    "a dedicated test suite (<code>test_no_leakage.py</code>) asserts no click ever appears in a "
    "user's history before it happened, run against the real downloaded data, not synthetic fixtures "
    "alone. One documented data limitation, not a workaround for convenience: MIND provides no "
    "per-click timestamps, only order, so synthetic per-user timestamps are assigned that are "
    "guaranteed to sit strictly before that split's earliest real impression.", body))

doc = SimpleDocTemplate(
    str(OUT_PATH), pagesize=A4,
    leftMargin=0.62 * inch, rightMargin=0.62 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch,
    title="CS4.406 A1 Design Note", author="Shubham Paliwal",
)
doc.build(story)
print(f"wrote {OUT_PATH}")
