# Project PRD: Generative Foundation Model for City Infrastructure Named Bonzai

## 0. How to read this document (for Claude Code)

This is the foundational document for the project. Read it in full before making decisions. When in doubt, return to it.

**About the user:** The user is new to deep learning and generative modeling. They are technically capable but have not trained transformers, designed tokenizers, or run scaling experiments before. When you explain anything technical, use simple analogies and concrete examples. When you make a design choice, briefly explain why in plain language before showing code. Never assume background knowledge of attention, autoregressive generation, diffusion, state-space models, or training infrastructure — explain or link to a quick explanation each time these come up. Treat every technical conversation as an opportunity to teach.

**Working principles for this project:**

1. Lock decisions before coding. The biggest mistake on this project would be to write a thousand lines of training code based on an assumption that turns out wrong. Before writing non-trivial code, restate the assumption you're making and ask for confirmation if it's load-bearing.
2. Test-driven where it matters. The tokenizer, the boundary contract derivation, the cell-stitching protocol, the eval metrics — these need real tests. Use the superpowers skills/plugins ecosystem for test-driven workflows.
3. Reproducibility from day one. Every experiment must be re-runnable. Every model checkpoint must be traceable to a config file, a data snapshot, and a code commit. No notebook-driven results that can't be reproduced.
4. Small before big. Validate every component at the smallest scale that proves it works before scaling. A single tile before a hundred tiles. A 30M parameter model before a 300M one. A toy eval before the full eval suite.
5. Architectural humility. This document describes the current best plan based on extensive discussion, but reality will push back. When experiments disagree with this PRD, the experiments win. Update the PRD; don't override the data.

## 1. What we are building

A generative foundation model that produces structured city geometry — roads, buildings, land use polygons, and points of interest — from a small set of conditioning parameters. The output is standards-compliant GeoJSON that can be loaded directly into game engines, simulators, and GIS tools.

The analogy to keep in mind: this is **GPT for cities**. Just as a language model trained on text learns the structure of language and can generate any text, this model trained on global map data learns the structure of urban form and can generate any plausible city or fragment thereof.

The primary intended applications are autonomous vehicle simulation environments, robotics training, defense and security simulations, and procedural content generation for games. The shared requirement across these is geometric precision, controllability, scalability, and reproducibility — not photorealism.

## 2. What we are not building

To prevent scope drift, this project explicitly is not:

- A 3D city renderer or visualization tool. The output is 2D geometry with attributes; downstream tools handle rendering.
- A photorealistic city image generator. We generate structured data, not pixels.
- A real-time editor or planning interface. Those could come later as products; the model itself is the deliverable here.
- A tool that requires manual curation of training data. We pull from public sources (OpenStreetMap, Overture Maps) and process them automatically.
- A replacement for any specific software (Esri ArcGIS, Autodesk products, etc.). We produce data those tools can ingest.

## 3. Why standard approaches fail and why ours is different

The naive approach to generating cities is to treat them as images: render the map as a raster, train a diffusion model on those rasters, sample new rasters, vectorize them. This was the user's previous architecture and it failed for principled reasons that should not be repeated:

Cities are sparse, high-frequency, geometric data — roads are single-pixel-wide topological structures, building corners are discontinuities, blocks have hard tiling constraints. Continuous-valued operations like VAE compression and diffusion in latent space are systematically biased against exactly these features. The training pipeline produces blurry rasters; the writer that translates rasters back to vectors has no signal to read; the output collapses to grammar-valid but geometrically meaningless GeoJSON.

The lesson learned: cities are not images. Treating them as images destroys the very properties we need to preserve. The architecture for this project rejects the raster intermediate entirely. We work in tokens — discrete symbolic representations of geometric primitives — from end to end. Tokens preserve right angles, exact connectivity, and discrete categories perfectly because they are discrete by construction.

## 4. The architecture in plain language

The model is a two-stage autoregressive transformer with no raster, no VAE, and no diffusion.

**Stage one: the macro planner.** A transformer reads conditioning parameters (region, density, era, style, scale) and emits a coarse plan for the entire tile. The plan describes the city at low resolution: zoning of each cell, density of each cell, the skeleton of major roads, and the boundary contracts for every shared edge between cells. The plan is small (a few thousand tokens) and globally coherent. Analogy: an urban planner sketching on graph paper before any block-level design.

**Stage two: the micro generator.** For each cell in the plan (a 2km × 2km tile is typically subdivided into an 8×8 grid of 250m × 250m cells), a second transformer takes the macro plan plus the cell's boundary contracts and generates the full geometric detail inside the cell. Roads, buildings, POIs — all in the relative-geometry token language. Cells generate in parallel. Each cell is independently generatable; the boundary contracts ensure they fit together. Analogy: many block-level architects working simultaneously, each handed the same neighborhood plan and a specification of what crosses the edges of their block.

**Stitching.** After all cells generate, a deterministic stitching pass resolves shared boundaries — the road that exits cell 1 on its east edge connects to the road entering cell 2 on its west edge, because both were derived from the same boundary contract. Stitching is a rules-based step, not a model.

The key innovation is the boundary contract. The macro plan does not just specify zoning — it specifies, for every shared cell edge, exactly what crosses it: each road's class, position, and width. Both neighboring cells receive this contract as conditioning and are trained to honor it. This is what enables parallel generation without losing coherence.

## 5. The data pipeline

The model trains on global map data, primarily Overture Maps (a cleaned, schema-normalized successor to raw OpenStreetMap that combines contributions from OSM, Meta, Microsoft, and TomTom). Overture is released monthly as GeoParquet on public S3, queryable directly with DuckDB.

The pipeline has the following stages:

**Stage one: vocabulary analysis.** Before any tokenization, run frequency analyses on every categorical field in Overture to determine empirical category distributions. Categories with fewer than 10,000 global instances bucket up to their parent category or are dropped. The goal is the maximum semantic granularity where every category has enough training examples to be reliably learned. Expected vocabulary size after curation: roughly 80–150 place categories, 8–15 building classes, 6–10 road classes, 10–15 land use classes. Combined with the geometric vocabulary, the total token vocabulary is in the 1,000–3,000 range.

**Stage two: tile extraction.** The world is divided into 2km × 2km tiles, each subdivided into an 8×8 cell grid (250m cells). For each tile, the pipeline extracts all features within the bounding box, reprojects to a local metric coordinate system anchored at the tile origin, and clips features at cell boundaries (recording crossings).

**Stage three: macro plan derivation.** For each tile, derive: per-cell zoning (dominant land use), per-cell density (binned building footprint area), the major road skeleton, and the complete set of boundary contracts. Boundary contracts list every feature that crosses each cell edge: feature class, crossing position quantized to the local grid, width or extent class.

**Stage four: micro token derivation.** For each cell, emit the token sequence in the relative-geometry language proven in the user's prior single-city proof of concept. Tokens describe shapes as sequences of moves from anchor points, preserving the abstract concept of geometric shapes that the model has shown it can learn (95% perfect right angles on the PoC). Features that cross cell boundaries terminate exactly at the crossing position recorded in the boundary contract.

**Stage five: consistency validation.** An automated validator confirms, for every processed tile, that the macro plan matches the underlying geometry, that every boundary contract corresponds to actual cell tokens, and that token sequences are decodable to valid GeoJSON. Failures are quarantined for inspection.

The complete output of the pipeline is, per tile, a directory containing: conditioning vector, macro plan tokens, per-cell micro tokens, per-cell boundary contracts, and provenance metadata.

## 6. The four candidate architectures for the bake-off

Before committing to a specific architecture for the large-scale training run, we test four candidates at small scale and pick the winner based on scaling curve extrapolation.

**Candidate one: pure autoregressive transformer.** A single transformer generates the entire tile as one long token sequence, left to right. This is the baseline — the simplest possible approach. It serves to calibrate the others. Likely to suffer on long sequences but useful as a control.

**Candidate two: hierarchical autoregressive transformer.** Two transformers, macro planner and micro generator, as described in section 4. This is the primary candidate. It builds directly on the user's prior PoC, matches the data pipeline design, and is well-supported by existing tooling.

**Candidate three: hierarchical autoregressive with Mamba-transformer hybrid backbone.** Same architectural shape as candidate two, but the transformer layers inside each model are replaced with an interleaved stack of Mamba (state-space model) and transformer layers, typically at a ratio of seven Mamba layers per one transformer layer (Jamba-style). Mamba is a linear-cost sequence model that complements transformer attention well on long, locally-coherent sequences. Mentioned often in design conversations because: our sequences are long, our data is locally coherent after Hilbert sorting, and inference speed will matter for downstream applications. Available off the shelf via the `mamba-ssm` PyPI package and Hugging Face's `transformers` library.

**Candidate four: discrete diffusion over tokens.** Instead of autoregressive generation, the model iteratively refines a fully-masked token sequence over many denoising steps. Each step, the model sees the entire partially-noised sequence (bidirectional context) and predicts which tokens to commit. Active research area as of 2024–2025 (SEDD, MDLM). Worth piloting at small scale because it has different parallelism properties than AR and might outperform on global coherence.

The bake-off methodology: train each candidate at three parameter scales (30M, 100M, 300M) with identical compute budgets, identical tokenizer, identical training data, identical evaluation. Fit scaling curves of evaluation loss versus compute. Extrapolate to the target production scale. The architecture with the best projected performance at scale wins. This methodology is borrowed from the language modeling community's scaling-laws work (Kaplan, Hoffmann/Chinchilla) and has the property of being right by construction — if you can't extrapolate the curve, you can't make a confident scaling decision.

## 7. Positional encoding decisions

This is a subtle but important design choice. The token sequence is one-dimensional, but the underlying geometry is two-dimensional. Standard rotary positional embeddings (RoPE) treat position as a single index, which discards the 2D structure.

The plan is to augment standard positional embeddings with explicit 2D positional signals — either as a second RoPE axis, learned 2D Fourier features, or ALiBi-style distance biases. The exact choice will be determined by the phase-three ablation sweep. The point worth recording in this PRD is that we are committed to *some* form of 2D-aware positional encoding rather than treating the sequence as pure 1D. This matters because Hilbert-sorted spatial tokens that are 50 positions apart in sequence may be one step apart in physical space, and the model should be able to recognize that.

## 8. Conditioning vocabulary

Rather than encoding rich regional or stylistic detail in every micro token, we use a small conditioning vector at the start of every training sequence. This is the equivalent of a system prompt for the model — it tells the model "produce a city of this type" and the model uses that signal to specialize its generation.

The conditioning vocabulary includes: morphology class (European-organic, North-American-grid, Asian-megacity, Middle-Eastern, African, Latin-American, Southeast-Asian-informal, etc.), country and admin region, era class (pre-industrial, industrial, post-1950, contemporary), climate zone, coastal/inland/riverside flag, population density bucket, and a deterministic seed. These are a few dozen tokens at most, prepended to every training sequence and to every inference prompt.

The advantage of this design over fine-grained per-token semantic conditioning is that the per-token vocabulary stays small (preserving learnability of rare tokens) while the conditioning surface stays expressive (enabling controllable generation). This is the same pattern used in text-to-image models.

## 9. Evaluation

A model is only as good as its evaluation. We define evaluation up front, before any training runs, so we don't unconsciously bias toward whatever metric makes our current model look good.

The evaluation suite has several layers:

**Geometric validity metrics.** Percentage of building polygons with corner angles within tolerance of 90°. Percentage of roads connecting to other roads (not dead-ending). Percentage of features with topologically valid geometry (no self-intersections, no overlap with disallowed feature classes). The user's PoC achieved 95% perfect right angles; this is the bar to beat.

**Statistical realism metrics.** Distribution of building sizes, road segment lengths, block sizes, and POI density compared to real cities of similar conditioning. Wasserstein distance or Kolmogorov-Smirnov distance against ground truth distributions.

**Topological metrics.** Road network connectivity (largest connected component as fraction of total road length). Intersection density per km². Betweenness centrality distribution.

**Generalization metrics.** Train on cities from regions A, B, C; evaluate on region D. Generated cities should match D's training distribution when conditioned on D. This is the central test of whether the model has learned generalizable urban form or only memorized training cities. A model that fails this test is not a foundation model.

**Conditioning compliance.** Evaluate on combinations of conditioning the model has not seen in training. Does conditioning on "European, low-density, coastal" produce something different from "Asian, high-density, inland"? Measure the model's ability to honor each conditioning dimension.

**Simulation viability.** Generated cities can be loaded into a basic simulator (CARLA-like) without crashes from invalid geometry. Roads have drivable lane structure. This is the actual user-facing quality bar for the intended applications.

All eval metrics are computed automatically against a held-out set of real cities. The held-out set is locked at the start of the project and never used for training.

## 10. Compute, infrastructure, and the Leonardo allocation

The user has 1,200 node-hours on Leonardo (EuroHPC), where each node has four A100 GPUs. This is approximately 4,800 GPU-hours, which is enough for serious experimental work but not unlimited. Budget allocation:

- Data pipeline (mostly CPU, but some GPU for embedding-based deduplication if needed): ~100 GPU-hours
- Phase 2 bake-off (12 runs across 4 architectures × 3 scales): ~1,500 GPU-hours
- Phase 3 ablations on the winning architecture: ~600 GPU-hours
- Production scale training run (after winner is identified): ~2,000 GPU-hours
- Reserve for re-runs, debugging, and surprises: ~600 GPU-hours

This budget assumes models in the 30M–3B parameter range. If experiments suggest the optimal model is much larger, we revisit the plan; we do not silently exceed the compute budget.

All training uses PyTorch with mixed precision (bfloat16 on A100), gradient checkpointing where memory-bound, and `torch.compile` for kernel fusion. Distributed training across nodes uses PyTorch's FSDP or DeepSpeed depending on scale. Mamba layers come from the `mamba-ssm` package (Tri Dao's official implementation). Transformer layers are standard PyTorch. Discrete diffusion uses the SEDD or MDLM implementations adapted for our tokenizer.

Checkpointing is mandatory every 30 minutes during training. Every checkpoint records: model weights, optimizer state, RNG state, training step, evaluation metrics, and a hash of the data shard processed. A training run that crashes is resumable from the last checkpoint with bit-identical continuation.

## 11. Project phases and milestones

The work is structured in phases. Each phase has a clear deliverable and a decision point at the end.

**Phase 0: Setup (week 1).** Repository scaffolding. Reproducible environment (conda or uv lock files, pinned versions). Test infrastructure. CI for the data pipeline. Decision point: can a new contributor clone the repo and reproduce a tiny end-to-end run? If not, fix the setup before proceeding.

**Phase 1: Data pipeline (weeks 2–3).** Build the full data pipeline from Overture Parquet to per-tile token sequences. Validate on a single hand-picked tile end to end. Run the vocabulary frequency analysis. Lock the tokenizer schema. Build the consistency validator. Generate the eval set. Decision point: can we tokenize 100 tiles correctly, with the validator passing on all of them, and can we round-trip back to GeoJSON?

**Phase 2: Architecture bake-off (weeks 4–6).** Train all four candidate architectures at three scales each on a representative training set of 10,000–50,000 tiles. Compute evaluation metrics for every checkpoint. Fit scaling curves. Decision point: which architecture has the best projected performance at production scale? Document the decision and its evidence.

**Phase 3: Refinement of the winner (weeks 7–8).** Ablation sweeps on the winning architecture: positional encoding variants, cell size, macro plan resolution, boundary contract format. Decision point: lock the final architecture configuration. Estimate the compute required for the production run.

**Phase 4: Production training (weeks 9–12).** Train the final model at production scale on the full curated training set. Continuous evaluation against the held-out set. Decision point: does the production model meet the evaluation bars set in section 9?

**Phase 5: Eval and release prep (week 13+).** Comprehensive evaluation. Documentation. Inference API. Sample generations. Public report.

This timeline is aggressive. It will probably slip. That is fine. Slippage is allowed; skipping decision points is not.

## 12. What the codebase should look like

A general shape, not prescriptive in detail:

```
city-foundation-model/
├── PRD.md                          # this document
├── README.md                       # quick-start
├── pyproject.toml / requirements   # pinned dependencies
├── configs/                        # YAML/TOML configs for every experiment
│   ├── data/
│   ├── tokenizer/
│   ├── architectures/
│   └── experiments/
├── data/                           # raw and processed (gitignored; large)
├── src/cfm/                        # main package
│   ├── tokenizer/                  # vocabulary, encoding, decoding
│   ├── data/                       # Overture loading, tile extraction, validation
│   ├── models/
│   │   ├── pure_ar.py              # candidate 1
│   │   ├── hierarchical_ar.py      # candidate 2
│   │   ├── mamba_hybrid.py         # candidate 3
│   │   └── discrete_diffusion.py   # candidate 4
│   ├── training/                   # training loop, optimization, distributed
│   ├── eval/                       # metrics and eval orchestration
│   └── inference/                  # generation, decoding, GeoJSON export
├── tests/                          # tests for everything in src/
├── scripts/                        # CLI entry points (launch training, eval, etc.)
├── notebooks/                      # exploration only, never authoritative
└── reports/                        # write-ups of phase decisions
```

Every experiment is a config file. Every result is a directory under `reports/` with the config, the metrics, the checkpoint reference, and a short prose summary of what was learned. The config plus the code commit fully determines the experiment.

## 13. Risks and what to do about them

A few risks worth naming up front because they could derail the project:

**The tokenizer turns out wrong.** If after phase 1 the vocabulary is misaligned with model needs (too long-tail, too coarse, missing essential distinctions), every downstream experiment is contaminated. Mitigation: heavy validation of the tokenizer before any model training. Round-trip every tile (raw → tokens → raw) and verify reconstruction quality.

**The macro/micro split fails to deliver coherence.** If boundary contracts don't ensure cells stitch correctly, the parallel architecture loses its main advantage. Mitigation: implement and test cell stitching extensively at phase 1, before any training. Build it as a deterministic algorithm with a test suite that covers edge cases (empty cells, cells with rivers, cells with major roads, cells at tile borders).

**Scaling curves don't separate the architectures.** If all four candidates perform similarly at small scale, we cannot confidently pick a winner. Mitigation: in that case, pick the simplest (likely hierarchical transformer) and focus compute on data and scale rather than architecture.

**Generalization fails.** If the model memorizes training cities and produces gibberish on held-out regions, it is not a foundation model. Mitigation: aggressive regional augmentation (rotation, reflection), region-conditioning that's enforced via classifier-free guidance during training, and continuous monitoring of held-out region metrics throughout training.

**Compute runs out before the production run.** The 4,800 GPU-hour budget is tight. Mitigation: strict per-phase budgets, kill underperforming runs early, and prefer smaller-but-converged models over larger-but-undertrained ones.

## 14. Working with Claude Code on this project

A few standing instructions specific to this collaboration:

When implementing something new, propose the design in plain language first, then ask for confirmation before writing code. The user is learning along with the implementation and needs the conceptual hook before the syntax.

When using technical terminology, briefly define or analogize the first time it appears in any conversation. "Autoregressive (meaning the model generates one token at a time, each token conditioned on the previous ones, like writing a sentence word by word)." Don't assume the term has been internalized just because it appeared earlier.

When you encounter a design choice not specified in this PRD, flag it explicitly. Don't silently pick. Either ask the user, or pick and explain the choice in a comment so it can be reviewed later.

Use the superpowers skills and plugins for test-driven workflows. Write tests for tokenizer round-trips, for boundary contract validation, for eval metric computations, for everything that has a clear input-output specification.

Keep `PRD.md` updated. When the project's understanding evolves, edit this document and commit the change. The PRD is the single source of truth; if it disagrees with the code or with reality, fix the PRD.

Default to simplicity. The user has been burned once by an over-engineered raster pipeline. When in doubt, choose the architecturally simpler option even if it seems less impressive. Working models beat clever models.

When the user expresses frustration or confusion, slow down. Re-explain in plainer terms. Ask what specifically is unclear. Do not paper over confusion with technical confidence — that's how people end up six months into a project they don't understand.

## 15. Definitions for quick reference

**Autoregressive (AR):** A generation approach where the model produces one token at a time, each conditioned on all previous tokens. Like writing a sentence word by word.

**Transformer:** A neural network architecture where every token attends to every other token. Powerful but expensive: cost grows quadratically with sequence length.

**Mamba / state-space model (SSM):** A neural network architecture where the model maintains a running summary as it processes tokens sequentially. Cheaper than transformers (linear cost with sequence length) but with a lossy summary.

**Hybrid (Mamba-Transformer):** An architecture that alternates Mamba layers and transformer layers, getting the speed of Mamba on most layers and the precision of transformers on a few key layers.

**Tokenization:** Converting raw data (images, maps, text) into a sequence of discrete symbols (tokens) that a model can process. Our tokens describe geometric primitives.

**Macro plan:** The output of the first stage of our model — a coarse description of the city at the cell-grid level: zoning, density, road skeleton, and boundary contracts.

**Boundary contract:** A specification of exactly what crosses each cell edge — which roads, at what positions, of what classes. Both adjacent cells receive this as conditioning, ensuring they generate consistent geometry across the boundary.

**Cell:** A subdivision of a tile, typically 250m × 250m. The unit of micro-scale generation.

**Tile:** The top-level unit of generation, typically 2km × 2km. Composed of an 8×8 grid of cells.

**Scaling curve:** A plot of model performance versus model size or compute. Used to extrapolate how performance will change with more resources, enabling principled decisions about which architecture to scale.

**FSDP / DeepSpeed:** Frameworks for distributed training that allow a single model to be split across multiple GPUs.

**Overture Maps:** A public, cleaned, schema-normalized successor to OpenStreetMap, released monthly as GeoParquet on public S3. Our primary training data source.

**GeoJSON:** A standard JSON-based format for geographic data. Roads as LineStrings, buildings and land use as Polygons, points of interest as Points. Our model's output format.
