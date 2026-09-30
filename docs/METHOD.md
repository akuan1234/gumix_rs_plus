# Semantic Prototype Enrichment

The default strategy is `gumix_rs_plus`. The previous name,
`stage2_blendalias06_tiny_compound05_pool10`, remains a compatibility alias
for the same strategy and coefficients.

| Prompt group | Templates | Weight |
| --- | ---: | ---: |
| Generic | 80 | 1.000 |
| Remote-sensing view | 10 | 0.040 |
| Segmentation | 10 | 0.025 |
| Context | 10 | 0.015 |
| Attribute | 10 | 0.015 |
| Infrastructure | 6 | 0.010 |

Sentence embeddings are L2-normalized and averaged within each group, followed
by normalization. Each canonical representation is blended with its mean alias
representation and normalized. The weighted group representations are summed
and normalized to form the final prototype.

The alias coefficient is 0.06, with fixed overrides of 0.05 for `road flooded`
and `building non-flooded`, and 0.10 for `pool`. Names are normalized and
deduplicated; at most four expressions, including the canonical name, are used.
Queries without aliases retain the canonical representation.

The strategy uses fixed prompts and coefficients. It does not train the text
encoder or change the geometry-uncertainty routing rule. Available controls are
`imagenet` (80 templates), `rs` (5 templates) and `mixed` (85 templates).

The prompt coefficients and dataset thresholds were selected with evaluation
feedback and then fixed for the reported comparisons. Frozen inference does not
imply that these settings were chosen without evaluation feedback.

The visual path uses the last four DINOv3 layers for geometry-dependent feature
fusion and selects one scale per pixel from `[1.0, 1.5]`. Semantic Prototype
Enrichment changes the shared class prototypes; geometry fusion and scale
routing retain the inherited visual computation. See the
[inference settings and implementation limits](EVALUATION.md) for the fixed
inference parameters and region-ID behavior.
