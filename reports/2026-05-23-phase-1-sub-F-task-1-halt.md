**Halt 1: BP1 vocab floor elbow**

Status: `DONE_WITH_CONCERNS` pending reviewer approval for elbow + exception list + X-threshold lock.

Pre-dispatch audits:
- Audit step 1 passed: `https://taginfo.openstreetmap.org/api/4/key/values?key=highway&page=1&rp=999&sortname=count&sortorder=desc` returned JSON with a `data` array and `value` / `count` fields in the first row. No taginfo API shape drift; no 7th cascade.
- Audit step 2 passed: `https://wiki.openstreetmap.org/w/api.php?action=query&prop=revisions&titles=Map_features&rvprop=content|ids&rvslots=main&format=json&formatversion=2` returned `query.pages[0].revisions[0].slots.main.content` and `revid`. No MediaWiki API shape drift; no 7th cascade.
- Audit step 3 passed: [src/cfm/data/sub_c/enums.py](/Users/umaraslam/Projects/Bonzai-OSM/src/cfm/data/sub_c/enums.py:1) still defines `FEATURE_CLASS: dict[int, str] = {0: "road", 1: "building", 2: "poi", 3: "base"}`. No sub-C enum expansion; no 7th cascade.

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

**Revised proposed elbow:**
- Granularity level: `L1+L2-mixed` (pure L1 is no longer proposed because it gives the cascade-#4 Singapore X paired check no `(key, value)` slots to bite on).
- Candidate mixed row A: `F = F_l1 = 0.0005953467798612618`; `207` slots before X-threshold exceptions; admits all 28 L1 must-appears and 21/56 L2 must-appear pairs.
- Candidate mixed row B: must-appears-only mixed lock; `F_needed = F_l2 = 9.95794913319044e-08`; `84` slots before X-threshold exceptions; admits all 28 L1 must-appears and all 56 L2 highway/building must-appear pairs with no discretionary slots.
- No cascade #7 is opened against cascade #4 by this augmentation; the proposed direction keeps cascade #4 load-bearing by retaining L2 highway/building pair slots.

**Proposed X-threshold (cascade #4 scope: highway + building only):**
- Candidate A: Singapore-elbow-derived `1.4337986487881535e-06`
- Candidate B: median must-appear frequency `0.004748741124786364`
- Scope note: POI + base deferred per spec `§12 #11`
- Paired structural check framing: any `(highway, value)` / `(building, value)` pair with Singapore frequency `>= X` must appear above `F` in the future `semantic_vocab.yaml` lock; this dispatch intentionally stops before writing that lock.
- Building pagination confirmation: `8,767` building value rows present in the taginfo CSV (`>= 8000` safeguard satisfied)

Singapore X candidate B pass-list (`22` pairs; `18` wiki-L2, `4` non-wiki exceptions):
- `building=B__UNK__` — count `301418`, freq `0.43217272112`, wiki_L2=`no`
- `building=residential` — count `40814`, freq `0.0585190580516`, wiki_L2=`yes`
- `building=house` — count `20369`, freq `0.0292050446772`, wiki_L2=`yes`
- `building=industrial` — count `5028`, freq `0.00720913960611`, wiki_L2=`no`
- `building=apartments` — count `4560`, freq `0.00653812183847`, wiki_L2=`yes`
- `building=commercial` — count `4525`, freq `0.00648793888577`, wiki_L2=`yes`
- `building=terrace` — count `3312`, freq `0.00474874112479`, wiki_L2=`yes`
- `highway=service` — count `100085`, freq `0.143501737764`, wiki_L2=`yes`
- `highway=footway` — count `78891`, freq `0.113113809202`, wiki_L2=`yes`
- `highway=residential` — count `35873`, freq `0.051434658928`, wiki_L2=`yes`
- `highway=primary` — count `14041`, freq `0.0201319668276`, wiki_L2=`yes`
- `highway=unknown` — count `9748`, freq `0.0139766692284`, wiki_L2=`no`
- `highway=tertiary` — count `9611`, freq `0.0137802388135`, wiki_L2=`yes`
- `highway=secondary` — count `9230`, freq `0.0132339615283`, wiki_L2=`yes`
- `highway=cycleway` — count `7876`, freq `0.0112925981579`, wiki_L2=`yes`
- `highway=steps` — count `7573`, freq `0.0108581571673`, wiki_L2=`yes`
- `highway=unclassified` — count `7463`, freq `0.0107004393159`, wiki_L2=`yes`
- `highway=motorway` — count `4929`, freq `0.00706719353988`, wiki_L2=`yes`
- `highway=trunk` — count `4675`, freq `0.00670300868308`, wiki_L2=`yes`
- `highway=subway` — count `4314`, freq `0.00618540737087`, wiki_L2=`no`
- `highway=path` — count `3491`, freq `0.00500539108292`, wiki_L2=`yes`
- `highway=track` — count `3444`, freq `0.00493800254643`, wiki_L2=`yes`

Singapore X candidate A pass-list has `80` total pairs (`35` wiki-L2, `45` non-wiki exceptions). It is the candidate B list above plus these `58` additional pairs:
- `building=parking` — count `2157`, freq `0.00309270368544`, wiki_L2=`no`
- `building=semidetached_house` — count `2054`, freq `0.00294502242461`, wiki_L2=`yes`
- `building=roof` — count `1872`, freq `0.00268407107053`, wiki_L2=`no`
- `building=detached` — count `1594`, freq `0.00228547504617`, wiki_L2=`yes`
- `building=retail` — count `1476`, freq `0.00211628680561`, wiki_L2=`no`
- `building=school` — count `1274`, freq `0.00182665947856`, wiki_L2=`no`
- `building=public` — count `634`, freq `0.000909028343332`, wiki_L2=`yes`
- `building=train_station` — count `466`, freq `0.000668150170335`, wiki_L2=`yes`
- `building=office` — count `447`, freq `0.000640907996008`, wiki_L2=`yes`
- `building=storage_tank` — count `441`, freq `0.000632305204116`, wiki_L2=`no`
- `building=dormitory` — count `376`, freq `0.000539108291944`, wiki_L2=`yes`
- `building=hotel` — count `320`, freq `0.000458815567612`, wiki_L2=`yes`
- `building=service` — count `246`, freq `0.000352714467602`, wiki_L2=`yes`
- `building=warehouse` — count `206`, freq `0.00029536252165`, wiki_L2=`no`
- `building=university` — count `182`, freq `0.000260951354079`, wiki_L2=`no`
- `building=temple` — count `178`, freq `0.000255216159484`, wiki_L2=`no`
- `building=hospital` — count `152`, freq `0.000217937394616`, wiki_L2=`no`
- `building=church` — count `131`, freq `0.000187827622991`, wiki_L2=`no`
- `building=garage` — count `127`, freq `0.000182092428396`, wiki_L2=`no`
- `building=kindergarten` — count `79`, freq `0.000113270093254`, wiki_L2=`no`
- `building=transportation` — count `78`, freq `0.000111836294605`, wiki_L2=`no`
- `building=hangar` — count `73`, freq `0.000104667301362`, wiki_L2=`yes`
- `building=bungalow` — count `65`, freq `9.31969121712e-05`, wiki_L2=`yes`
- `building=college` — count `59`, freq `8.45941202785e-05`, wiki_L2=`no`
- `building=greenhouse` — count `50`, freq `7.16899324394e-05`, wiki_L2=`no`
- `building=shed` — count `49`, freq `7.02561337906e-05`, wiki_L2=`yes`
- `building=grandstand` — count `43`, freq `6.16533418979e-05`, wiki_L2=`no`
- `building=mosque` — count `41`, freq `5.87857446003e-05`, wiki_L2=`no`
- `building=hut` — count `38`, freq `5.44843486539e-05`, wiki_L2=`no`
- `building=pavilion` — count `25`, freq `3.58449662197e-05`, wiki_L2=`no`
- `building=sports_centre` — count `24`, freq `3.44111675709e-05`, wiki_L2=`no`
- `building=government` — count `22`, freq `3.15435702733e-05`, wiki_L2=`no`
- `building=library` — count `22`, freq `3.15435702733e-05`, wiki_L2=`yes`
- `building=toilets` — count `19`, freq `2.7242174327e-05`, wiki_L2=`no`
- `building=stadium` — count `15`, freq `2.15069797318e-05`, wiki_L2=`no`
- `building=guardhouse` — count `14`, freq `2.0073181083e-05`, wiki_L2=`no`
- `building=civic` — count `12`, freq `1.72055837855e-05`, wiki_L2=`no`
- `building=fire_station` — count `12`, freq `1.72055837855e-05`, wiki_L2=`no`
- `building=farm_auxiliary` — count `11`, freq `1.57717851367e-05`, wiki_L2=`yes`
- `building=religious` — count `11`, freq `1.57717851367e-05`, wiki_L2=`no`
- `building=stable` — count `10`, freq `1.43379864879e-05`, wiki_L2=`no`
- `building=garages` — count `6`, freq `8.60279189273e-06`, wiki_L2=`no`
- `building=sports_hall` — count `5`, freq `7.16899324394e-06`, wiki_L2=`no`
- `building=stilt_house` — count `5`, freq `7.16899324394e-06`, wiki_L2=`yes`
- `building=bunker` — count `4`, freq `5.73519459515e-06`, wiki_L2=`no`
- `building=chapel` — count `4`, freq `5.73519459515e-06`, wiki_L2=`no`
- `building=manufacture` — count `4`, freq `5.73519459515e-06`, wiki_L2=`no`
- `building=post_office` — count `4`, freq `5.73519459515e-06`, wiki_L2=`no`
- `building=shrine` — count `4`, freq `5.73519459515e-06`, wiki_L2=`no`
- `building=boathouse` — count `3`, freq `4.30139594636e-06`, wiki_L2=`no`
- `building=synagogue` — count `2`, freq `2.86759729758e-06`, wiki_L2=`no`
- `building=bridge_structure` — count `1`, freq `1.43379864879e-06`, wiki_L2=`no`
- `building=carport` — count `1`, freq `1.43379864879e-06`, wiki_L2=`no`
- `building=kiosk` — count `1`, freq `1.43379864879e-06`, wiki_L2=`no`
- `building=monastery` — count `1`, freq `1.43379864879e-06`, wiki_L2=`no`
- `building=supermarket` — count `1`, freq `1.43379864879e-06`, wiki_L2=`yes`
- `highway=pedestrian` — count `579`, freq `0.000830169417648`, wiki_L2=`yes`
- `highway=living_street` — count `448`, freq `0.000642341794657`, wiki_L2=`yes`

**Cascade documentation (mandatory per spec §13.5):**
- Cascade #4 outcome: Singapore X scope = highway + building. POI/base deferred to sub-F-v2.
- Cascade #5 outcome: L1 corrected to 28 keys; L3 deferred entirely.
- Cascade #6 outcome: taginfo `rp=1000` rejected; implementation uses `rp=999`, paginates `building` only, and documents non-scope L1 value-tail cap per spec `§12 #12`.
- `§13.5` protocol-v2 candidates surfaced:
  1. transitive-documentation citing
  2. hand-enumeration with complete-count assertion
  3. reviewer-supplied lists as untrusted input
  4. dispatch prompt audits reuse implementation call/code path
  5. exact-parameter upstream diagnostics
  6. reviewer-supplied parameter values as untrusted input

**Self-review checks:**
- `_status` in [configs/sub_f/vocab_floor_analysis.yaml](/Users/umaraslam/Projects/Bonzai-OSM/configs/sub_f/vocab_floor_analysis.yaml) is `PROPOSED`
- `wiki_l1_must_appears` length = `28`
- `wiki_l2_highway_count = 23`
- `wiki_l2_building_count = 33`
- `wiki_l3_status = "deferred per spec §12 #10"`
- `curve` has `3` rows
- [tests/data/sub_f/test_vocab.py](/Users/umaraslam/Projects/Bonzai-OSM/tests/data/sub_f/test_vocab.py) passes `8/8`
- `configs/sub_f/semantic_vocab.yaml` does not exist

**§10.5 telemetry:**
- Implementer-time-to-data-surface: approximately `10` wall-clock minutes from dispatch start to Halt 1 report commit.
