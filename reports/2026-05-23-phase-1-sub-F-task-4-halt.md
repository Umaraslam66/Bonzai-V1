**Halt 3: BP4 unknown family + sentinel inventory**

Status: `DONE` - Halt 3 approved; Task 4 closed.

**Enumerated `<unknown_*>` slots:**
1. `<unknown_aerialway>` - key `aerialway` - locked ID `200`
2. `<unknown_aeroway>` - key `aeroway` - locked ID `201`
3. `<unknown_amenity>` - key `amenity` - locked ID `202`
4. `<unknown_barrier>` - key `barrier` - locked ID `203`
5. `<unknown_boundary>` - key `boundary` - locked ID `204`
6. `<unknown_building>` - key `building` - locked ID `205`
7. `<unknown_craft>` - key `craft` - locked ID `206`
8. `<unknown_emergency>` - key `emergency` - locked ID `207`
9. `<unknown_geological>` - key `geological` - locked ID `208`
10. `<unknown_healthcare>` - key `healthcare` - locked ID `209`
11. `<unknown_highway>` - key `highway` - locked ID `210`
12. `<unknown_historic>` - key `historic` - locked ID `211`
13. `<unknown_landuse>` - key `landuse` - locked ID `212`
14. `<unknown_leisure>` - key `leisure` - locked ID `213`
15. `<unknown_man_made>` - key `man_made` - locked ID `214`
16. `<unknown_military>` - key `military` - locked ID `215`
17. `<unknown_natural>` - key `natural` - locked ID `216`
18. `<unknown_office>` - key `office` - locked ID `217`
19. `<unknown_place>` - key `place` - locked ID `218`
20. `<unknown_power>` - key `power` - locked ID `219`
21. `<unknown_public_transport>` - key `public_transport` - locked ID `220`
22. `<unknown_railway>` - key `railway` - locked ID `221`
23. `<unknown_route>` - key `route` - locked ID `222`
24. `<unknown_shop>` - key `shop` - locked ID `223`
25. `<unknown_telecom>` - key `telecom` - locked ID `224`
26. `<unknown_tourism>` - key `tourism` - locked ID `225`
27. `<unknown_water>` - key `water` - locked ID `226`
28. `<unknown_waterway>` - key `waterway` - locked ID `227`

**Singapore occurrence table:**

| token | key | locked semantic-pair coverage | real OSM below F | sub-C sentinels | total |
|---|---:|---:|---:|---:|---:|
| `<unknown_aerialway>` | `aerialway` | 0 | 0 | 0 | 0 |
| `<unknown_aeroway>` | `aeroway` | 0 | 0 | 0 | 0 |
| `<unknown_amenity>` | `amenity` | 0 | 0 | 0 | 0 |
| `<unknown_barrier>` | `barrier` | 0 | 0 | 0 | 0 |
| `<unknown_boundary>` | `boundary` | 0 | 0 | 0 | 0 |
| `<unknown_building>` | `building` | 93759 | 0 | 301418 | 301418 |
| `<unknown_craft>` | `craft` | 0 | 0 | 0 | 0 |
| `<unknown_emergency>` | `emergency` | 0 | 0 | 0 | 0 |
| `<unknown_geological>` | `geological` | 0 | 0 | 0 | 0 |
| `<unknown_healthcare>` | `healthcare` | 0 | 0 | 0 | 0 |
| `<unknown_highway>` | `highway` | 292523 | 0 | 9748 | 9748 |
| `<unknown_historic>` | `historic` | 0 | 0 | 0 | 0 |
| `<unknown_landuse>` | `landuse` | 0 | 0 | 0 | 0 |
| `<unknown_leisure>` | `leisure` | 0 | 0 | 0 | 0 |
| `<unknown_man_made>` | `man_made` | 0 | 0 | 0 | 0 |
| `<unknown_military>` | `military` | 0 | 0 | 0 | 0 |
| `<unknown_natural>` | `natural` | 0 | 0 | 0 | 0 |
| `<unknown_office>` | `office` | 0 | 0 | 0 | 0 |
| `<unknown_place>` | `place` | 0 | 0 | 0 | 0 |
| `<unknown_power>` | `power` | 0 | 0 | 0 | 0 |
| `<unknown_public_transport>` | `public_transport` | 0 | 0 | 0 | 0 |
| `<unknown_railway>` | `railway` | 0 | 0 | 0 | 0 |
| `<unknown_route>` | `route` | 0 | 0 | 0 | 0 |
| `<unknown_shop>` | `shop` | 0 | 0 | 0 | 0 |
| `<unknown_telecom>` | `telecom` | 0 | 0 | 0 | 0 |
| `<unknown_tourism>` | `tourism` | 0 | 0 | 0 | 0 |
| `<unknown_water>` | `water` | 0 | 0 | 0 | 0 |
| `<unknown_waterway>` | `waterway` | 0 | 0 | 0 | 0 |

**Over-firing / zero-firing locked table:**

| token | numerator | denominator | ratio | over-firing | zero-firing | rationale |
|---|---:|---:|---:|---|---|---|
| `<unknown_aerialway>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=aerialway in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_aeroway>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=aeroway in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_amenity>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=amenity in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_barrier>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=barrier in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_boundary>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=boundary in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_building>` | 301418 | 93759 | 3.214817 | `True` | `False` | Flag when unknown total / locked semantic-pair Singapore coverage >= 10% for key=building. |
| `<unknown_craft>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=craft in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_emergency>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=emergency in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_geological>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=geological in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_healthcare>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=healthcare in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_highway>` | 9748 | 292523 | 0.033324 | `False` | `False` | Flag when unknown total / locked semantic-pair Singapore coverage >= 10% for key=highway. |
| `<unknown_historic>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=historic in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_landuse>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=landuse in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_leisure>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=leisure in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_man_made>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=man_made in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_military>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=military in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_natural>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=natural in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_office>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=office in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_place>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=place in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_power>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=power in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_public_transport>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=public_transport in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_railway>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=railway in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_route>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=route in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_shop>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=shop in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_telecom>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=telecom in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_tourism>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=tourism in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_water>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=water in sub-C v1 mapping; ratio left null and over-firing remains false. |
| `<unknown_waterway>` | 0 | 0 | n/a | `False` | `True` | No scoped Singapore semantic-pair denominator for key=waterway in sub-C v1 mapping; ratio left null and over-firing remains false. |

**Halt 3 continuation addendum: zero-firing slot retention**

26 zero-firing slots are scope-of-coverage-zero, not OSM-real-zero. They preserve v1 IDs against multi-region unknown expansion at sub-F-v2 per cascade #4 deferral.

**Halt 3 continuation addendum: B__UNK__ / highway=unknown decomposition**

- `building=B__UNK__`: raw-cache join missing count `0` across `301418` rows; raw-class top-20 coverage `100.0000%`.
- Classification: root cause (b), real OSM long-tail / upstream under-typed source data; no BP1 cascade #8 and no sub-C-v2 candidate from this decomposition. Raw Overture buildings.class is NULL for the sentinel rows, so there are no recoverable wiki/semantic building values hidden behind B__UNK__ in class_raw.

`building=B__UNK__` raw `buildings.class` top values:

| raw value | count | fraction |
|---|---:|---:|
| `<NULL>` | 301418 | 100.0000% |

`building=B__UNK__` raw `buildings.subtype` top values:

| raw value | count | fraction |
|---|---:|---:|
| `<NULL>` | 299237 | 99.2764% |
| `civic` | 1018 | 0.3377% |
| `religious` | 497 | 0.1649% |
| `commercial` | 369 | 0.1224% |
| `entertainment` | 146 | 0.0484% |
| `education` | 70 | 0.0232% |
| `medical` | 64 | 0.0212% |
| `transportation` | 13 | 0.0043% |
| `residential` | 4 | 0.0013% |

- `highway=unknown`: raw-cache join missing count `0` across `9748` rows; raw-class top-20 coverage `100.0000%`.
- Classification: root cause (b), real OSM long-tail / literal upstream `unknown`; no BP1 cascade #8 and no sub-C-v2 candidate from this decomposition. Raw Overture transportation.class is literal 'unknown' for these rows, not a hidden wiki/semantic highway value.

`highway=unknown` raw `transportation.class` top values:

| raw value | count | fraction |
|---|---:|---:|
| `unknown` | 9748 | 100.0000% |

`highway=unknown` raw `transportation.subtype` top values:

| raw value | count | fraction |
|---|---:|---:|
| `road` | 8226 | 84.3865% |
| `rail` | 1522 | 15.6135% |

**Halt 3 ID namespace anchor:**

- BP1 semantic family: `0..199` (`127` used, `72` reserved for v2 semantic growth) - LOCKED.
- BP4 unknown family: `200..255` (`28` used at `200..227`, `28` reserved at `228..255`) - LOCKED.
- Dataloader-side sentinels: `256..299` with `<pad>=256`, `<eos>=257`, `<bos>=258`, `<cell_start>=259`, `<cell_end>=260`; these are not on-disk sub-F vocab tokens - LOCKED.
- BP2 encoding primitives: placeholder block `300..1499`; values lock at Task 2 halt - PLACEHOLDER.
- BP7 boundary-ref: placeholder block `1500..1599`; values lock at Task 7 halt, `8` expected used and `92` reserved - PLACEHOLDER.

**`sentinel_inventory.yaml` post-N/dataloader reservation:**

- `_status`: `LOCKED`
- Dataloader sentinel reservations are marked `on_disk: false`.
- BP2/BP7 remain explicit placeholder blocks in the namespace anchor.

**Placeholder-block caveat required at Halt 3:**

- BP2 (`300..1499`) and BP7 (`1500..1599`) are PLACEHOLDER blocks.
- Their final sizes are empirically locked at Tasks 2 and 7 halts respectively.
- If Task 2 encoding-primitive count lands above `1200` or Task 7 boundary-ref count above `100`, BP2/BP7 blocks slide.
- Only BP1 + BP4 + dataloader sentinel IDs are LOCKED at Halt 3 approval.

**§10.5 telemetry:**

- Implementer-time-to-data-surface: approximately `30` wall-clock minutes from Task 4 start to Halt 3 report generation.
- Report generated at: `2026-05-27T16:19:26Z`.
