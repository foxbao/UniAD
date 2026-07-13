# Documentation index

This directory separates current decisions from research notes and historical
experiment detail. Do not read every document to recover the project state.

## Read first

1. [`mapfuse_planning_analysis.md`](mapfuse_planning_analysis.md) - current
   promoted checkpoint, active experiment, evaluation contract, and next route.
2. [`llm_assisted_map_planning_research.md`](llm_assisted_map_planning_research.md)
   - LLM/VLA literature, proposed Planning IR, and phased promotion gates.
3. [`planning_ir_schema.md`](planning_ir_schema.md) - implemented P0 schema,
   GT-leak boundary, audit workflow, and executable commands.

## Historical evidence

- [`archive/mapfuse_planning_experiment_history.md`](archive/mapfuse_planning_experiment_history.md)
  preserves the complete v1-v6/A-D chronology, intermediate hypotheses,
  rejected variants, old commands, and detailed tables. It is evidence, not the
  current roadmap.

## Current decision order

1. Keep D2 medium as the formal reference and D2 full as the global-best
   initialization; full still misses the strict Turning gate.
2. Run one D2.1 partial-unfreeze experiment on candidate representation plus
   cost head, preserving the exact fallback.
3. Implement D3 top-K set-aware reranking after the D2.1 control.
4. Validate the positive Planning-IR P0 signal on a holdout with shuffled and
   equal-input non-LLM controls before any runtime LLM conditioning.
5. Attempt hybrid diffusion/VLA only if proposal coverage becomes the measured
   bottleneck.

## Scope note

Some working trees also contain older motion-turn-aware, VLM-caption,
LLMBridge, environment, and session-resume documents. They belong to separate
work lines and are not authoritative for the current map-planning branch. Code
comments may still reference those paths, so this cleanup does not rename or
rewrite them.

Patent-related directories remain excluded from version control.
