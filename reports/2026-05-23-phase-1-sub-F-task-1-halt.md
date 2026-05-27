**Halt 1: BP1 vocab floor elbow**

Status: `DONE` with F-elbow and X-threshold locked by reviewer; `semantic_vocab.yaml` locked in post-Halt-1 Steps 10-11.

Pre-dispatch audits:
- Audit step 1 passed: `https://taginfo.openstreetmap.org/api/4/key/values?key=highway&page=1&rp=999&sortname=count&sortorder=desc` returned JSON with a `data` array and `value` / `count` fields in the first row. No taginfo API shape drift.
- Audit step 2 passed: `https://wiki.openstreetmap.org/w/api.php?action=query&prop=revisions&titles=Map_features&rvprop=content|ids&rvslots=main&format=json&formatversion=2` returned `query.pages[0].revisions[0].slots.main.content` and `revid`. No MediaWiki API shape drift.
- Audit step 3 passed: [src/cfm/data/sub_c/enums.py](/Users/umaraslam/Projects/Bonzai-OSM/src/cfm/data/sub_c/enums.py:1) still defines `FEATURE_CLASS: dict[int, str] = {0: "road", 1: "building", 2: "poi", 3: "base"}`. No sub-C enum expansion. Cascade #7 surfaced later from sub-C normalized value semantics, not enum shape.

**Snapshot artifacts:**
- taginfo CSV:
  - path: [configs/sub_f/taginfo/2026-04-15.0.csv](/Users/umaraslam/Projects/Bonzai-OSM/configs/sub_f/taginfo/2026-04-15.0.csv)
  - row count: 150,879 total CSV lines
  - first 5 rows verbatim:
    ```text
    key,value,count_all,count_ways,count_nodes,count_relations,row_type,parent_key
    building,,700650203,698470868,995082,1184253,key,
    building,yes,557917852,0,0,0,value,building
    building,house,65940990,0,0,0,value,building
    building,residential,16104287,0,0,0,value,building
    ```
  - file size: 8,050,930 bytes
  - sha256: `e4b2659482f5df90261db584414504889903f939b45362284f47bde24ce67184`
- wiki Map_features:
  - wikitext path: [configs/sub_f/wiki_map_features/2026-04-15.0.wikitext](/Users/umaraslam/Projects/Bonzai-OSM/configs/sub_f/wiki_map_features/2026-04-15.0.wikitext)
  - revision_id: `2876652`
  - wikitext byte count: 2,915
  - sha256: `18c79db75286bb78b4dd93908525600469ea682e1ad435a45f6800859d8cda22`

**Marginal-cost curve (L1 full + L2 highway+building; L3 deferred):**
- Pure L1 row: 28 L1 must-appear keys, `F_min = 0.0005953467798612618`, `vocab_size = 186` (`28` must-appear + `158` discretionary L1 key slots).
- Pure L2 row: 56 highway+building primary pairs, `F_min = 9.95794913319044e-08`, `vocab_size = 56` (`56` must-appear + `0` discretionary L2 pair slots). This row is diagnostic only; pure L2 would drop L1 key coverage.
- L1+L2-mixed at `F_l1 = 0.0005953467798612618`: `186` L1 key slots + `21` L2 pair slots whose frequency is `>= F_l1` = `207` total slots. Delta vs pure L1 = `+21` slots, not `-130`.
- L1+L2-mixed must-appears-only row: `F_needed = F_l2 = 9.95794913319044e-08`; explicit lock of `28` L1 must-appear keys + `56` L2 highway/building must-appear pairs + `0` discretionary slots = `84` total slots. Note: applying `F_l2` as a plain threshold without the must-appears-only rule would admit `200` L1 key rows + `56` L2 pairs = `256` slots.
- L3 row: deferred per spec `§12 #10`.

L2 pairs admitted in the L1+L2-mixed `F_l1` row (`21`):
- `building=yes`, `building=house`, `building=residential`, `building=detached`, `building=apartments`, `building=shed`, `building=farm_auxiliary`, `building=semidetached_house`
- `highway=residential`, `highway=service`, `highway=footway`, `highway=track`, `highway=unclassified`, `highway=path`, `highway=tertiary`, `highway=secondary`, `highway=primary`, `highway=living_street`, `highway=cycleway`, `highway=steps`, `highway=trunk`

**F_l2 diagnostic:**
- `F_l2` argmin pair: `building=tree_house`
- taginfo count: `263`
- fraction: `9.95794913319044e-08`
- Upstream wiki verification: `Map_features` transcludes `Template:Building typology`; live MediaWiki fetch of `Template:Building typology` revision `2952985` contains the row `building | tree_house` with `Tag:building=tree_house`. This is a rare canonical building typology, not a non-primary spillover. No Gate 6 enumeration defect surfaced from this argmin.

**Reviewer F-elbow lock:**
- Granularity level: `L1+L2-mixed`.
- Locked F: `F_l2 = 9.95794913319044e-08`.
- Exception list for F-elbow: `[]`.
- Slot count before X-threshold exceptions: `84` (`28` L1 semantic categories + `56` L2 highway/building primary pairs).
- Discretionary L1 key rows admitted at `F_l1` are not admitted by default; future additions are scoped expansions at Task 4 or later.

**X-threshold status (cascade #4 + #7 scope: highway + building real OSM values only):**
- X lock: `LOCKED_BY_REVIEWER` to Candidate A' at `2.5887822885870944e-06`.
- Sentinel filter applied before X derivation: `building=B__UNK__` count `301418` and `highway=unknown` count `9748` removed (`311166` features total).
- Filtered Singapore denominator: `386282`.
- Candidate A': Singapore-elbow-derived `2.5887822885870944e-06` (`LOCKED`).
- Candidate B': median must-appear frequency `0.008574046939800456`.
- Scope note: POI + base deferred per spec `§12 #11`; sub-C unknown sentinels map to BP4 `<unknown_*>`, not BP1 semantic slots.
- Paired structural check framing: any real `(highway, value)` / `(building, value)` pair with filtered Singapore frequency `>= X` must appear above `F` in the locked `semantic_vocab.yaml`; the post-Halt-1 lock admits the 35 wiki-L2 pass-list pairs through the 56 L2 must-appears and the 43 non-wiki Singapore-empirical pairs as first-class slots.
- Building pagination confirmation: `8,767` building value rows present in the taginfo CSV (`>= 8000` safeguard satisfied).

Filtered Singapore X candidate B' pass-list (`20` pairs; `18` wiki-L2, `2` non-wiki exceptions):
- `building=residential` - count `40814`, freq `0.105658560326`, wiki_L2=`yes`
- `building=house` - count `20369`, freq `0.0527309064362`, wiki_L2=`yes`
- `building=industrial` - count `5028`, freq `0.013016397347`, wiki_L2=`no`
- `building=apartments` - count `4560`, freq `0.011804847236`, wiki_L2=`yes`
- `building=commercial` - count `4525`, freq `0.0117142398559`, wiki_L2=`yes`
- `building=terrace` - count `3312`, freq `0.0085740469398`, wiki_L2=`yes`
- `highway=service` - count `100085`, freq `0.259098275353`, wiki_L2=`yes`
- `highway=footway` - count `78891`, freq `0.204231623529`, wiki_L2=`yes`
- `highway=residential` - count `35873`, freq `0.0928673870385`, wiki_L2=`yes`
- `highway=primary` - count `14041`, freq `0.0363490921141`, wiki_L2=`yes`
- `highway=tertiary` - count `9611`, freq `0.0248807865756`, wiki_L2=`yes`
- `highway=secondary` - count `9230`, freq `0.0238944605237`, wiki_L2=`yes`
- `highway=cycleway` - count `7876`, freq `0.0203892493049`, wiki_L2=`yes`
- `highway=steps` - count `7573`, freq `0.0196048482715`, wiki_L2=`yes`
- `highway=unclassified` - count `7463`, freq `0.0193200822197`, wiki_L2=`yes`
- `highway=motorway` - count `4929`, freq `0.0127601079004`, wiki_L2=`yes`
- `highway=trunk` - count `4675`, freq `0.0121025571991`, wiki_L2=`yes`
- `highway=subway` - count `4314`, freq `0.011168006793`, wiki_L2=`no`
- `highway=path` - count `3491`, freq `0.00903743896946`, wiki_L2=`yes`
- `highway=track` - count `3444`, freq `0.00891576620189`, wiki_L2=`yes`

Filtered Singapore X candidate A' pass-list has `78` total pairs (`35` wiki-L2, `43` non-wiki exceptions). It is the candidate B' list above plus these `58` additional pairs:
- `building=parking` - count `2157`, freq `0.00558400339648`, wiki_L2=`no`
- `building=semidetached_house` - count `2054`, freq `0.00531735882076`, wiki_L2=`yes`
- `building=roof` - count `1872`, freq `0.00484620044424`, wiki_L2=`no`
- `building=detached` - count `1594`, freq `0.00412651896801`, wiki_L2=`yes`
- `building=retail` - count `1476`, freq `0.00382104265795`, wiki_L2=`no`
- `building=school` - count `1274`, freq `0.00329810863566`, wiki_L2=`no`
- `building=public` - count `634`, freq `0.00164128797096`, wiki_L2=`yes`
- `building=train_station` - count `466`, freq `0.00120637254648`, wiki_L2=`yes`
- `building=office` - count `447`, freq `0.001157185683`, wiki_L2=`yes`
- `building=storage_tank` - count `441`, freq `0.00114165298927`, wiki_L2=`no`
- `building=dormitory` - count `376`, freq `0.000973382140509`, wiki_L2=`yes`
- `building=hotel` - count `320`, freq `0.000828410332348`, wiki_L2=`yes`
- `building=service` - count `246`, freq `0.000636840442992`, wiki_L2=`yes`
- `building=warehouse` - count `206`, freq `0.000533289151449`, wiki_L2=`no`
- `building=university` - count `182`, freq `0.000471158376523`, wiki_L2=`no`
- `building=temple` - count `178`, freq `0.000460803247369`, wiki_L2=`no`
- `building=hospital` - count `152`, freq `0.000393494907865`, wiki_L2=`no`
- `building=church` - count `131`, freq `0.000339130479805`, wiki_L2=`no`
- `building=garage` - count `127`, freq `0.000328775350651`, wiki_L2=`no`
- `building=kindergarten` - count `79`, freq `0.000204513800798`, wiki_L2=`no`
- `building=transportation` - count `78`, freq `0.00020192501851`, wiki_L2=`no`
- `building=hangar` - count `73`, freq `0.000188981107067`, wiki_L2=`yes`
- `building=bungalow` - count `65`, freq `0.000168270848758`, wiki_L2=`yes`
- `building=college` - count `59`, freq `0.000152738155027`, wiki_L2=`no`
- `building=greenhouse` - count `50`, freq `0.000129439114429`, wiki_L2=`no`
- `building=shed` - count `49`, freq `0.000126850332141`, wiki_L2=`yes`
- `building=grandstand` - count `43`, freq `0.000111317638409`, wiki_L2=`no`
- `building=mosque` - count `41`, freq `0.000106140073832`, wiki_L2=`no`
- `building=hut` - count `38`, freq `9.83737269663e-05`, wiki_L2=`no`
- `building=pavilion` - count `25`, freq `6.47195572147e-05`, wiki_L2=`no`
- `building=sports_centre` - count `24`, freq `6.21307749261e-05`, wiki_L2=`no`
- `building=government` - count `22`, freq `5.69532103489e-05`, wiki_L2=`no`
- `building=library` - count `22`, freq `5.69532103489e-05`, wiki_L2=`yes`
- `building=toilets` - count `19`, freq `4.91868634832e-05`, wiki_L2=`no`
- `building=stadium` - count `15`, freq `3.88317343288e-05`, wiki_L2=`no`
- `building=guardhouse` - count `14`, freq `3.62429520402e-05`, wiki_L2=`no`
- `building=civic` - count `12`, freq `3.1065387463e-05`, wiki_L2=`no`
- `building=fire_station` - count `12`, freq `3.1065387463e-05`, wiki_L2=`no`
- `building=farm_auxiliary` - count `11`, freq `2.84766051745e-05`, wiki_L2=`yes`
- `building=religious` - count `11`, freq `2.84766051745e-05`, wiki_L2=`no`
- `building=stable` - count `10`, freq `2.58878228859e-05`, wiki_L2=`no`
- `building=garages` - count `6`, freq `1.55326937315e-05`, wiki_L2=`no`
- `building=sports_hall` - count `5`, freq `1.29439114429e-05`, wiki_L2=`no`
- `building=stilt_house` - count `5`, freq `1.29439114429e-05`, wiki_L2=`yes`
- `building=bunker` - count `4`, freq `1.03551291543e-05`, wiki_L2=`no`
- `building=chapel` - count `4`, freq `1.03551291543e-05`, wiki_L2=`no`
- `building=manufacture` - count `4`, freq `1.03551291543e-05`, wiki_L2=`no`
- `building=post_office` - count `4`, freq `1.03551291543e-05`, wiki_L2=`no`
- `building=shrine` - count `4`, freq `1.03551291543e-05`, wiki_L2=`no`
- `building=boathouse` - count `3`, freq `7.76634686576e-06`, wiki_L2=`no`
- `building=synagogue` - count `2`, freq `5.17756457717e-06`, wiki_L2=`no`
- `building=bridge_structure` - count `1`, freq `2.58878228859e-06`, wiki_L2=`no`
- `building=carport` - count `1`, freq `2.58878228859e-06`, wiki_L2=`no`
- `building=kiosk` - count `1`, freq `2.58878228859e-06`, wiki_L2=`no`
- `building=monastery` - count `1`, freq `2.58878228859e-06`, wiki_L2=`no`
- `building=supermarket` - count `1`, freq `2.58878228859e-06`, wiki_L2=`yes`
- `highway=pedestrian` - count `579`, freq `0.00149890494509`, wiki_L2=`yes`
- `highway=living_street` - count `448`, freq `0.00115977446529`, wiki_L2=`yes`

**Cascade documentation (mandatory per spec §13.5):**
- Cascade #4 outcome: Singapore X scope = highway + building. POI/base deferred to sub-F-v2.
- Cascade #5 outcome: L1 corrected to 28 keys; L3 deferred entirely.
- Cascade #6 outcome: taginfo `rp=1000` rejected; implementation uses `rp=999`, paginates `building` only, and documents non-scope L1 value-tail cap per spec `§12 #12`.
- Cascade #7 outcome: sub-C normalization sentinels are filtered before Singapore X derivation; encoder maps them to BP4 `<unknown_*>`, not dedicated BP1 semantic slots.
- `§13.5` protocol-v2 candidates surfaced:
  1. transitive-documentation citing
  2. hand-enumeration with complete-count assertion
  3. reviewer-supplied lists as untrusted input
  4. dispatch prompt audits reuse implementation call/code path
  5. exact-parameter upstream diagnostics
  6. reviewer-supplied parameter values as untrusted input
  7. Singapore-frequency pass-lists filter upstream normalization sentinels

**Self-review checks:**
- `_status` in [configs/sub_f/vocab_floor_analysis.yaml](/Users/umaraslam/Projects/Bonzai-OSM/configs/sub_f/vocab_floor_analysis.yaml) is `PROPOSED`
- `wiki_l1_must_appears` length = `28`
- `wiki_l2_highway_count = 23`
- `wiki_l2_building_count = 33`
- `wiki_l3_status = "deferred per spec §12 #10"`
- `curve` has `3` rows
- `proposed_elbow.status = "LOCKED_BY_REVIEWER_FOR_F_ELBOW; X-threshold pending"` remains historical in the PROPOSED floor-analysis artifact; the approved X lock is recorded in `locked_semantic_vocab`
- `proposed_x_threshold.sentinel_filter.status = "applied before X derivation per cascade #7"`
- [tests/data/sub_f/test_vocab.py](/Users/umaraslam/Projects/Bonzai-OSM/tests/data/sub_f/test_vocab.py) passes `9/9`
- [configs/sub_f/semantic_vocab.yaml](/Users/umaraslam/Projects/Bonzai-OSM/configs/sub_f/semantic_vocab.yaml) exists with `_status: LOCKED` and `127` first-class semantic slots

**§10.5 telemetry:**
- Implementer-time-to-data-surface: approximately `10` wall-clock minutes from dispatch start to Halt 1 report commit.

**Post-Halt-1 lock (Steps 10-11):**
- Timestamp: `2026-05-27T12:41:19Z`.
- Reviewer-approved F lock: `9.95794913319044e-08`.
- Reviewer-approved X lock: `2.5887822885870944e-06` (Candidate A').
- Granularity level: `L1+L2-mixed`.
- F-elbow exception list: `[]`.
- Semantic vocab lock: [configs/sub_f/semantic_vocab.yaml](/Users/umaraslam/Projects/Bonzai-OSM/configs/sub_f/semantic_vocab.yaml) with `127` slots (`28` L1 must-appear keys + `56` wiki-L2 highway/building pairs + `43` non-wiki Singapore-empirical Candidate A' pairs).
- Task 1 close status: `DONE`.
