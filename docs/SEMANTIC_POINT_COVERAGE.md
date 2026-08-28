# Semantic Point Coverage

Semantic Point Coverage measures whether a generated script covers evaluation references derived from official question-answer annotations. It is reported only as an external evaluation metric and is not part of candidate selection.

For a generated script, STARS joins the `narration`, `on_screen_text`, and `salient_point` fields from all timeline segments. The frozen SigLIP2 text tower embeds this text and each semantic-point reference. A reference is covered when their cosine similarity is at least `0.50`. The sample score is the fraction of covered references.

Each annotation row must contain a `semantic_points` list and provenance identifying the reference protocol as `official_qa_evidence_and_answer_v2`. The deterministic builder creates two references for every official question-answer pair:

- `Evidence context: <question>`
- `Expected answer: <answer>`

When several question-answer rows refer to the same video, all pairs are retained in the video-level record. Dataset questions, answers, captions, subtitles, transcripts, reference descriptions, and semantic-point references are excluded from generator and reward inputs.

Use `scripts/build_semantic_point_manifests.py` to construct the references and `scripts/audit_semantic_point_annotations.py` to verify their completeness and provenance.
