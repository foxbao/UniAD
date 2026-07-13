# Documentation index

This directory separates current decisions from research notes and historical
experiment detail. Do not read every document to recover the project state.

## Read first

1. [`mapfuse_planning_analysis.md`](mapfuse_planning_analysis.md) - current
   promoted checkpoint, active experiment, evaluation contract, and next route.
2. [`llm_assisted_map_planning_research.md`](llm_assisted_map_planning_research.md)
   - LLM/VLA literature, proposed Planning IR, and phased promotion gates.

## Historical evidence

- [`archive/mapfuse_planning_experiment_history.md`](archive/mapfuse_planning_experiment_history.md)
  preserves the complete v1-v6/A-D chronology, intermediate hypotheses,
  rejected variants, old commands, and detailed tables. It is evidence, not the
  current roadmap.

## Current decision order

1. Finish and evaluate D2 full; keep D2 medium as reference until then.
2. Implement D3 top-K set-aware reranking with exact fallback.
3. Run the LLM Planning-IR `P0` audit in parallel without modifying D3.
4. Promote LLM conditioning only after causal gain over an equal-input non-LLM
   baseline.
5. Attempt hybrid diffusion/VLA only if proposal coverage becomes the measured
   bottleneck.

## Scope note

Some working trees also contain older motion-turn-aware, VLM-caption,
LLMBridge, environment, and session-resume documents. They belong to separate
work lines and are not authoritative for the current map-planning branch. Code
comments may still reference those paths, so this cleanup does not rename or
rewrite them.

Patent-related directories remain excluded from version control.
