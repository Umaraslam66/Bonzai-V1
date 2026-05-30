# Phase 1 Sub-F Task 2 Halt 2 Surface

Status: DONE - Halt 2 approved; Task 2 closed

## Audit outcomes

- WKB writer audit: PASS; `src/cfm/data/sub_c/io.py` retains `byte_order=1`, `dump_wkb`, and shapely/WKB symbols.
- Singapore cache audit: PASS; tile count `494` from `tile=EPSG3414_*`, with WKB byte order verified as `1` before implementation.
- BP2 lock audit: PASS; block `300..1499` is LOCKED at Halt 2 approval with Task 8 writer sub-block ordering fixed.

## All-tile input inventory

- Tile count: 494
- Total feature count: 862436
- Geometry type feature counts: `{0: 149666, 1: 303302, 2: 399695, 4: 5690, 5: 4083}`
- Eligible primitive counts: `{'polylines': 315092, 'polygon_exterior_rings': 408700}`
- Measured sample counts: `{'polylines': 1000, 'polygon_exterior_rings': 1000}`

## Geometry primitive distributions

- Turn angles: `{'polylines_abs_deg': {'count': 794988, 'mean': 14.186135, 'p50': 8.006368, 'p95': 48.379857, 'p99': 90.585819, 'max': 179.998539}, 'polygon_exterior_rings_abs_deg': {'count': 2637483, 'mean': 83.511275, 'p50': 89.976487, 'p95': 94.829084, 'p99': 140.371925, 'max': 179.796385}}`
- Vertex spacing: `{'polylines': {'count': 1110080, 'mean': 19.652705, 'p50': 10.416614, 'p95': 68.368683, 'p99': 129.568841, 'max': 342.586384}, 'polygon_exterior_rings': {'count': 2637483, 'mean': 10.219272, 'p50': 6.418139, 'p95': 29.666283, 'p99': 65.007026, 'max': 344.087801}}`
- Building corner abs deviation from 90 deg: `{'count': 2551646, 'mean': 8.699538, 'p50': 0.255536, 'p95': 67.964586, 'p99': 89.902538, 'max': 90.0}`

## Joint candidate surface

| dirs | quantum_m | analytical_linf_m | linf_mean_m | linf_p50_m | linf_p95_m | linf_p99_m | linf_max_m | samples | skips |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 8 | 0.25 | 12.37087 | 6.202399 | 2.565138 | 26.263963 | 54.107252 | 93.40356 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 8 | 0.5 | 12.49587 | 6.220171 | 2.616825 | 26.265232 | 54.107252 | 93.40356 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 8 | 1.0 | 12.74587 | 6.284994 | 2.656481 | 26.265981 | 54.107252 | 93.40356 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 16 | 0.25 | 6.36789 | 2.977576 | 1.291279 | 12.186847 | 24.587543 | 36.489518 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 16 | 0.5 | 6.49289 | 3.010403 | 1.322873 | 12.28804 | 24.414844 | 36.489518 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 16 | 1.0 | 6.74289 | 3.102123 | 1.474671 | 12.254952 | 24.846901 | 36.489518 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 24 | 0.25 | 4.301838 | 2.019708 | 0.980589 | 7.462548 | 15.142919 | 26.029678 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 24 | 0.5 | 4.426838 | 2.056296 | 1.011991 | 7.585238 | 15.094348 | 26.029678 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |
| 24 | 1.0 | 4.676838 | 2.153339 | 1.115434 | 7.583544 | 15.126002 | 26.029678 | `{'polylines': 1000, 'polygon_exterior_rings': 1000}` | `{'polylines': 0, 'polygon_exterior_rings': 0}` |


Measured L_inf convention: polylines are open and include all original vertices; polygon exterior rings exclude the implicit closure vertex from original-vertex error measurement, and decoded closure is reconstructed.

## Singapore building right angles

- Definition: `abs(angle_deg - 90) <= 5`
- Total building polygon corner count: 2551646
- Input right-angle corner count: 2089236
- Fraction within +/-5 deg of 90: 0.81878
- Input caveat: Singapore building right-angle input fraction is 0.81878 (81.9%) within +/-5 deg of 90, not the POC 95% claim.
- Mean deviation is 8.7 deg and p95 is about 68 deg, so 5% of corners are substantially non-rectilinear.
- BP2 angular precision must handle the rectilinear majority and the curved/complex minority.
- 24 directions accepts known loss on the 5% non-rectilinear minority; this is a design tradeoff, not a bug.

| dirs | quantum_m | post_mean_deg | post_p50_deg | post_p95_deg | post_p99_deg | post_max_deg | change_p95_deg | change_p99_deg | corners | skips |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.25 | 1.293843 | 0.0 | 4.398705 | 45.0 | 90.0 | 4.751245 | 43.672223 | 2086045 | 3191 |
| 8 | 0.5 | 1.331283 | 0.0 | 5.350673 | 45.0 | 90.0 | 5.371478 | 43.900366 | 2082338 | 6898 |
| 8 | 1.0 | 1.428182 | 0.0 | 7.020364 | 45.0 | 90.0 | 6.871901 | 44.601859 | 2071223 | 18013 |
| 16 | 0.25 | 0.899663 | 0.0 | 3.814075 | 22.5 | 90.0 | 4.410996 | 22.15868 | 2086291 | 2945 |
| 16 | 0.5 | 0.969898 | 0.0 | 4.763642 | 22.5 | 90.0 | 4.853061 | 22.216266 | 2082827 | 6409 |
| 16 | 1.0 | 1.128016 | 0.0 | 6.340192 | 22.5 | 90.0 | 6.342622 | 22.551247 | 2072012 | 17224 |
| 24 | 0.25 | 0.816138 | 0.0 | 4.291384 | 15.0 | 90.0 | 4.603113 | 14.779519 | 2086425 | 2811 |
| 24 | 0.5 | 0.900961 | 0.0 | 5.201025 | 15.0 | 90.0 | 5.253961 | 14.96102 | 2083012 | 6224 |
| 24 | 1.0 | 1.075759 | 0.0 | 7.125016 | 19.775761 | 90.0 | 7.142823 | 19.830561 | 2072271 | 16965 |


## Halt 2 continuation addendum

### Item 1: L_inf decomposition at proposed (24, 0.5m)

- Sample count: 1000
- Skip count: 0
- Correlations: `{'total_length_m': {'pearson': 0.692405, 'spearman': 0.780424}, 'vertex_count': {'pearson': 0.258424, 'spearman': 0.429506}, 'max_abs_turn_angle_deg': {'pearson': 0.03913, 'spearman': 0.31168}, 'mean_vertex_spacing_m': {'pearson': 0.483501, 'spearman': 0.633627}}`
- Root-cause classification: `{'dominant_driver': 'total_length_m', 'dominant_abs_spearman': 0.780424, 'classification': 'length_correlated_threshold_chunking_lever', 'reviewer_action': 'Revisit chunking threshold or accumulated direction quantization before lock.'}`
- Threshold status: pending reviewer decision; no final L_inf lock is made in this continuation.

| rank | tile_id | source_feature_id | L_inf_m | length_m | vertices | max_turn_deg | mean_spacing_m |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | tile=EPSG3414_i5_j15 | b34f3fc5-9022-4f4e-a33e-5ac923fd1793 | 26.029678 | 252.011482 | 9 | 1.027644 | 31.501435 |
| 2 | tile=EPSG3414_i23_j24 | d0ab078f-aa1a-4b57-9a29-f9b235cf1bd7 | 24.298875 | 266.878671 | 2 | 0.0 | 266.878671 |
| 3 | tile=EPSG3414_i6_j21 | 2c07931b-dfee-4d81-8a35-b7c887054f71 | 22.649894 | 260.422332 | 5 | 22.671624 | 65.105583 |
| 4 | tile=EPSG3414_i17_j23 | dd34210c-c519-4fe4-b061-21bbf9ccffda | 21.846263 | 238.285686 | 3 | 0.05715 | 119.142843 |
| 5 | tile=EPSG3414_i12_j16 | 108b16b2-8972-4125-a388-aaceab254cd9 | 21.371145 | 242.036258 | 5 | 2.429904 | 60.509065 |
| 6 | tile=EPSG3414_i23_j24 | 09c9a03d-bf89-4bfc-851a-5418c4f71241 | 21.219808 | 183.750932 | 2 | 0.0 | 183.750932 |
| 7 | tile=EPSG3414_i3_j24 | 758b2c24-e4f6-46e6-b1a2-0ad48309fbd7 | 20.610246 | 252.475646 | 4 | 1.55866 | 84.158549 |
| 8 | tile=EPSG3414_i18_j25 | 67b5cb85-6011-456e-b097-ee23c721356a | 20.044217 | 309.826585 | 11 | 7.840014 | 30.982659 |
| 9 | tile=EPSG3414_i3_j14 | 2b248272-15df-48f0-9de0-eeef7ce374b4 | 19.059053 | 250.990122 | 9 | 22.445056 | 31.373765 |
| 10 | tile=EPSG3414_i18_j18 | 6abeea86-9d49-4cbd-934e-b85c9009121f | 19.003815 | 274.192272 | 8 | 11.621878 | 39.170325 |
| 11 | tile=EPSG3414_i7_j22 | cdc5419d-8b52-4fdf-bd89-e86a7a3d3110 | 18.900673 | 298.337679 | 8 | 24.886906 | 42.619668 |
| 12 | tile=EPSG3414_i2_j16 | fd67e7ee-e538-41ab-8826-2bd1d23755e1 | 18.445055 | 266.892358 | 9 | 4.640226 | 33.361545 |
| 13 | tile=EPSG3414_i12_j16 | 9e8e0c61-3ffb-47bc-a1a3-5b3b72ecac35 | 17.719391 | 199.541985 | 5 | 3.457914 | 49.885496 |
| 14 | tile=EPSG3414_i1_j24 | 428798ed-7fa9-4ab7-a351-5e9732b3c782 | 17.196418 | 198.488607 | 2 | 0.0 | 198.488607 |
| 15 | tile=EPSG3414_i7_j16 | edf473e9-3bef-41b2-8871-844d6dca345e | 16.075527 | 250.518434 | 6 | 1.836172 | 50.103687 |
| 16 | tile=EPSG3414_i2_j23 | 4db3702c-fc3e-4979-86ae-5a586e74d078 | 16.070101 | 280.813572 | 11 | 11.082661 | 28.081357 |
| 17 | tile=EPSG3414_i11_j14 | f1e5c003-a90f-4329-85ba-07ebb9e4761f | 15.695517 | 263.108478 | 15 | 92.673557 | 18.793463 |
| 18 | tile=EPSG3414_i6_j16 | 713cefef-20b3-43ac-9d63-f343e02a25c4 | 15.532888 | 146.764777 | 2 | 0.0 | 146.764777 |
| 19 | tile=EPSG3414_i2_j14 | 5fa84f4c-7950-41e4-af10-ab24365f1112 | 15.411587 | 149.226259 | 5 | 0.943589 | 37.306565 |
| 20 | tile=EPSG3414_i17_j23 | f1d6bb5f-d053-4b78-b377-d5df04337fa7 | 15.094038 | 179.778476 | 5 | 33.413617 | 44.944619 |
| 21 | tile=EPSG3414_i10_j21 | 6a6c5c59-734c-4340-b1aa-f019926155e5 | 15.037335 | 235.411048 | 7 | 13.076018 | 39.235175 |
| 22 | tile=EPSG3414_i4_j14 | c853932e-3be6-4b1a-b788-87ac827ca621 | 14.897807 | 201.159982 | 7 | 37.726252 | 33.526664 |
| 23 | tile=EPSG3414_i12_j24 | 9d94c80e-eae6-4cd7-a2f4-6ed960802cab | 14.767759 | 235.055138 | 8 | 1.89093 | 33.579305 |
| 24 | tile=EPSG3414_i16_j17 | 6c87c8fa-2ea7-4181-b6da-5e99c792d4fd | 14.582334 | 234.39405 | 3 | 1.523253 | 117.197025 |
| 25 | tile=EPSG3414_i16_j17 | 7368e695-840a-453b-bb51-76c20bd80e3c | 14.570683 | 236.526974 | 9 | 1.673811 | 29.565872 |
| 26 | tile=EPSG3414_i19_j19 | c60d6fc3-fad4-4d9f-b959-64795c57af89 | 14.490005 | 160.983683 | 8 | 1.68785 | 22.997669 |
| 27 | tile=EPSG3414_i7_j21 | 0a71cb64-241f-3c3d-836e-8847543614a1 | 14.240957 | 262.801489 | 3 | 24.244709 | 131.400745 |
| 28 | tile=EPSG3414_i18_j18 | c143084d-77a5-4fba-9bc5-f284a2d96ce1 | 13.949066 | 127.367653 | 3 | 1.061013 | 63.683826 |
| 29 | tile=EPSG3414_i17_j20 | 486b2fe2-e178-48fb-beac-91258fbd1f11 | 13.831497 | 263.278501 | 6 | 2.809574 | 52.6557 |
| 30 | tile=EPSG3414_i4_j16 | ea5319b1-6e70-49be-a6a4-98531fc0c4c7 | 13.789066 | 187.707731 | 3 | 0.147482 | 93.853866 |
| 31 | tile=EPSG3414_i23_j18 | ee960f52-2c3a-3d24-abab-bc2b7ed83edc | 13.376573 | 122.086079 | 2 | 0.0 | 122.086079 |
| 32 | tile=EPSG3414_i23_j18 | 01f31db0-ba0a-4266-a6a8-4f92be0f1424 | 13.282245 | 121.343951 | 2 | 0.0 | 121.343951 |
| 33 | tile=EPSG3414_i19_j17 | d5faf8fe-812d-4047-bbee-9f8761ed6579 | 13.090724 | 262.121045 | 5 | 13.876201 | 65.530261 |
| 34 | tile=EPSG3414_i15_j20 | 9d99707f-7321-4982-9fe1-a96cb460edf8 | 12.949211 | 179.78533 | 9 | 3.958228 | 22.473166 |
| 35 | tile=EPSG3414_i15_j15 | cbe05c34-8dd8-4019-bb54-b9b82a0adb39 | 12.938083 | 309.977978 | 3 | 4.677662 | 154.988989 |
| 36 | tile=EPSG3414_i16_j17 | 82c95689-71c4-4585-ac2c-adedfc22c404 | 12.172531 | 137.335959 | 5 | 5.597255 | 34.33399 |
| 37 | tile=EPSG3414_i13_j15 | 9e0e3ce4-502d-475a-b7d4-d8cc86b4576b | 12.058583 | 212.076851 | 10 | 10.420847 | 23.564095 |
| 38 | tile=EPSG3414_i14_j14 | 88656aaa-12f3-471c-91a0-a3657e8841c8 | 11.804143 | 261.803789 | 13 | 9.597295 | 21.816982 |
| 39 | tile=EPSG3414_i2_j22 | 08cac3fc-36d7-4de3-a42c-3cf2d19b2740 | 11.744796 | 135.544186 | 4 | 1.208594 | 45.181395 |
| 40 | tile=EPSG3414_i10_j19 | 3a7aff43-7650-4af4-965a-c582887b9037 | 11.676049 | 408.845289 | 12 | 46.81387 | 37.167754 |
| 41 | tile=EPSG3414_i5_j14 | ea31d863-770f-40bf-bc50-98c43b85079d | 11.635707 | 128.860266 | 2 | 0.0 | 128.860266 |
| 42 | tile=EPSG3414_i12_j15 | 198a832a-2107-47ba-ace5-dc5371c9e865 | 11.634339 | 219.273521 | 3 | 5.503537 | 109.636761 |
| 43 | tile=EPSG3414_i21_j21 | 325298e7-1cf9-45b0-8140-06cb5d848397 | 11.108489 | 267.922557 | 9 | 33.23351 | 33.49032 |
| 44 | tile=EPSG3414_i11_j24 | 270cd050-e578-42cd-ad95-cbb758f89c87 | 11.081864 | 170.372827 | 4 | 3.200939 | 56.790942 |
| 45 | tile=EPSG3414_i2_j16 | 24c6b227-3ded-4261-8c24-ea83ff96b1d0 | 10.993916 | 277.802649 | 13 | 66.368257 | 23.150221 |
| 46 | tile=EPSG3414_i15_j19 | fb8a213f-51f8-402e-bbdd-3565635a94b8 | 10.910065 | 123.311376 | 8 | 0.04417 | 17.615911 |
| 47 | tile=EPSG3414_i1_j21 | 814854a6-3985-4179-84c3-eb44333ded97 | 10.778479 | 145.884052 | 6 | 24.827071 | 29.17681 |
| 48 | tile=EPSG3414_i6_j14 | d18a4138-4f1f-4901-9479-43e6477028de | 10.749919 | 256.650856 | 8 | 20.926917 | 36.664408 |
| 49 | tile=EPSG3414_i12_j22 | 8ad5decb-e59b-4e5c-9d85-746ef5f8ffa7 | 10.573444 | 128.551023 | 2 | 0.0 | 128.551023 |
| 50 | tile=EPSG3414_i17_j16 | c7e915f1-d0e4-4cf4-8e74-d7e6b10ca720 | 10.453089 | 219.170994 | 10 | 19.716025 | 24.352333 |


### Item 2: right-angle catastrophic bucket decomposition at proposed (24, 0.5m)

- Measured right-angle corner count: 2083012
- Skip count: 6224
- Catastrophic count (>45 deg): 4517
- Catastrophic fraction: 0.00216849
- Classification: structural_encoding_bug_surface_for_plan_revision
- Angle threshold status: pending reviewer decision; no final angle lock is made in this continuation.

| bucket | count | fraction |
| --- | ---: | ---: |
| <5 deg | 1975732 | 0.94849766 |
| 5-15 deg | 65477 | 0.03143381 |
| 15-45 deg | 37286 | 0.01790004 |
| >45 deg | 4517 | 0.00216849 |


### Item 3: anchor proposal revision

- Proposed anchor scheme is revised to hierarchical.
- Rationale: flat consumes 1000/1200 BP2 placeholder slots (83%); hierarchical consumes 96/1200 (8%). The 14% mean sequence-length cost is bounded training-compute cost, while the 10x vocab namespace cost is permanent within phase. Hierarchical leaves headroom for sub-F-v2 anchor changes.

### Item 4: input characterization caveat

- Singapore building right-angle input fraction is 0.81878 (81.9%) within +/-5 deg of 90, not the POC 95% claim.
- Mean deviation is 8.7 deg; p95 is about 68 deg, so 5% of corners are substantially non-rectilinear.
- BP2 angular precision must handle rectilinear majority and curved/complex minority.
- 24 directions accepts known loss on the 5% non-rectilinear minority; this is design tradeoff, not bug.

### Continuation #2 Item A: L_inf chunking lever sweep

- Candidate: `{'direction_count': 24, 'magnitude_quantum_m': 0.5, 'anchor_scheme': 'hierarchical', 'geometry_class': 'polylines'}`
- Classification: chunking_is_no_op_on_test_sample
- Reviewer guidance: Chunk-as-lever hypothesis falsified: 32/24/16/12m chunk thresholds are identical to reported precision on this sample.
- Proposed chunk threshold metadata: 32 m
- Proposed L_inf threshold metadata: 15.1 m (PENDING_DIRECTION_SWEEP)

| chunk_m | mean_m | p50_m | p95_m | p99_m | max_m | samples | skips |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.872919 | 1.505613 | 10.342844 | 18.901704 | 26.029678 | 1000 | 0 |
| 24 | 2.872919 | 1.505613 | 10.342844 | 18.901704 | 26.029678 | 1000 | 0 |
| 16 | 2.872919 | 1.505613 | 10.342844 | 18.901704 | 26.029678 | 1000 | 0 |
| 12 | 2.872919 | 1.505613 | 10.342844 | 18.901704 | 26.029678 | 1000 | 0 |


### Continuation #3 Item A: direction-count sweep

- Candidate: `{'magnitude_quantum_m': 0.5, 'anchor_scheme': 'hierarchical', 'chunk_threshold_m': 32, 'geometry_class': 'polylines'}`
- Classification: forty_eight_dirs_restores_poc_target_band
- Reviewer guidance: 48 directions drop p95 into the 3-5m target band and fit BP2; propose direction_count=48.
- Proposed direction count: 48
- Proposed L_inf threshold: 4.8 m (PROPOSED_AFTER_DIRECTION_SWEEP)

| dirs | mean_m | p50_m | p95_m | p99_m | max_m | required BP2 slots | fits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 2.872919 | 1.505613 | 10.342844 | 18.901704 | 26.029678 | 185 | True |
| 32 | 2.084777 | 1.113633 | 7.869212 | 14.147772 | 21.94527 | 193 | True |
| 48 | 1.402802 | 0.767266 | 4.790393 | 8.335855 | 12.883383 | 209 | True |
| 72 | 0.985186 | 0.57306 | 3.339377 | 5.69697 | 7.857616 | 233 | True |

### Continuation #3 Item B: anchor verification / cross-measurement inconsistency

- Joint surface anchor scheme: anchor_scheme_independent
- Joint surface scope: combined_polylines_and_polygon_exterior_rings
- Lock-threshold surface anchor scheme: hierarchical
- Verified inconsistency cause: geometry_scope_difference_not_anchor_scheme
- Note: Analysis-local encode/decode uses vertex-anchor deltas; flat vs hierarchical anchor tokenization affects BP2 vocab/sequence accounting, not decoded geometry coordinates. The 7.59m vs 10.34m p95 mismatch comes from combined geometry-class joint-surface aggregation versus polyline-only lock-threshold measurement.
- Anchor tradeoff: Hierarchical saves namespace (96 anchor slots vs flat 1000) with a bounded sequence-length cost; L_inf lock-threshold surfaces are labeled hierarchical and do not rely on pre-revision joint-surface aggregate numbers.

### Continuation #2 Item B: right-angle catastrophic root-cause buckets

- Catastrophic corner count: 4517
- Root-cause classification: `{'classification': 'multiple_structural_hypotheses_triggered', 'reviewer_guidance': 'Catastrophic corners trigger multiple structural hypotheses; review anchor handling and direction-bin boundary/angle wrapping before angle lock.', 'triggered_hypotheses': ['possible_bp2_anchor_design_structural_issue_cascade_8_candidate', 'possible_direction_bin_boundary_or_angle_wrapping_bug', 'larger_perimeter_dominates_not_small_polygon_loss'], 'dominant_ring_position_bucket': {'bucket': 'position_0_or_1', 'count': 2681, 'fraction': 0.59353553}, 'dominant_input_deviation_bucket': {'bucket': '<1deg', 'count': 3608, 'fraction': 0.79876024}, 'dominant_perimeter_bucket': {'bucket': '>100m', 'count': 2714, 'fraction': 0.60084127}}`
- V1 classification: accepted_v1_known_loss_cascade_8_candidate_for_sub_f_v2
- Angle threshold proposal: 7.5 deg (PROPOSED_V1_KNOWN_LOSS_EXCLUDING_CATASTROPHIC), basis `non_catastrophic_post_deviation_p95`, catastrophic >45 deg excluded: True

Ring position buckets:

| bucket | count | fraction |
| --- | ---: | ---: |
| position_0_or_1 | 2681 | 0.59353553 |
| position_2_to_mid | 0 | 0.0 |
| position_mid_to_last | 1836 | 0.40646447 |


Input deviation buckets:

| bucket | count | fraction |
| --- | ---: | ---: |
| <1deg | 3608 | 0.79876024 |
| 1_to_3deg | 611 | 0.13526677 |
| 3_to_5deg | 298 | 0.06597299 |


Parent polygon perimeter buckets:

| bucket | count | fraction |
| --- | ---: | ---: |
| <10m | 53 | 0.01173345 |
| 10_to_30m | 385 | 0.08523356 |
| 30_to_100m | 1365 | 0.30219172 |
| >100m | 2714 | 0.60084127 |


### Continuation #3 Item D: protocol-v2 candidate 9 capture

- Protocol-v2 candidate (9th): when diagnostic measurement contradicts prior hypothesis classification, surface hypothesis falsified explicitly. Sub-F Task 2 Continuation #2 Item A: chunk-as-lever hypothesis was falsified by identical L_inf across 4 chunk values; classification should have been chunking_is_no_op_on_test_sample rather than default chunking retained.

## Collinearity admission threshold

- Candidate triples X: 320168
- Total polyline interior triples Y: 794945
- Weak empirical p95 basis: False
- Perpendicular deviation distribution: `{'count': 320168, 'mean': 0.230744, 'p50': 0.093474, 'p95': 0.928048, 'p99': 1.751071, 'max': 6.090856}`
- Fixed multiples: `{'1x_magnitude_quantum': 0.5, '2x_magnitude_quantum': 1.0}`
- Proposed method: `empirical_p95_perpendicular_deviation`
- Proposed threshold: 0.928048 m

Spec framing applied: collinearity admission threshold is the maximum perpendicular deviation from the straight line through neighboring decoded vertices.

## Anchor scheme comparison

| scheme | vocab size | tokens/anchor | mean seq/cell | p95 seq/cell | derivation |
| --- | ---: | ---: | ---: | ---: | --- |
| flat | 1000 | 2 | 638.968047 | 2024.0 | 2 * ceil(250 / magnitude_quantum_m) |
| hierarchical | 96 | 4 | 732.602199 | 2292.0 | (16 coarse + 32 fine) * 2 axes |

Boundary-reference overhead is out of scope for Task 2; Tasks 3/7 cover cross-cell overhead later.

## Locked BP2 primitive inputs

- Direction count: 48
- Magnitude quantum: 0.5 m
- Anchor scheme: hierarchical
- Chunk threshold: 32 m
- Round-trip L_inf threshold: 4.8 m (LOCKED_HALT_2_APPROVED)
- Round-trip 95th-percentile angle threshold: 7.5 deg (LOCKED_HALT_2_APPROVED; catastrophic >45 deg cases excluded from threshold basis)
- Collinearity admission perpendicular threshold: 0.928048 m
- Methodology preserved: deterministic seed `20260523`; polyline-only L_inf scope; non-catastrophic angle scope excludes post-roundtrip absolute deviation from 90 deg >45 deg.
- Rationale: Halt 2 approved lock. 48 directions restore polyline-only L_inf p95 to the target band while fitting BP2. Hierarchical anchors lock project design principle #1, cheap-to-keep and impossible-to-recover.

## BP2 locked fit and sub-blocks

- Block: 300..1499
- Used slots: 209
- Reserved/v2 headroom: 991
- Components: `{'anchor_vocab_size': 96, 'direction_vocab_size': 48, 'magnitude_vocab_size': 65}`
- Fits placeholder: True
- Sub-block order locked for Task 8 writer:
  - anchor: `300..395` (96)
  - direction: `396..443` (48)
  - magnitude: `444..508` (65)
  - BP2 reserved/v2 headroom: `509..1499` (991)
- BP7 remains PLACEHOLDER at `1500..1599`; Task 7 halt locks BP7.

## Halt 2 close addendum

- `configs/sub_f/encoding_primitives.yaml` is LOCKED by Halt 2 approval.
- `configs/sub_f/sentinel_inventory.yaml` transitions BP2 `300..1499` from PLACEHOLDER to LOCKED.
- Cascade #8 candidate for sub-F-v2: BP2 anchor + direction-bin alignment.
- Right-angle catastrophic known-loss accepted for v1: 4517 / 2.08M, approximately 0.22%; non-catastrophic angle threshold locks at 7.5 deg.
- Chunking-is-no-op classification retained: Continuation #2 Item A measured identical L_inf across 32/24/16/12m chunk thresholds.
- Protocol-v2 candidate #9 capture retained: diagnostic measurements that falsify prior hypotheses must explicitly surface `hypothesis falsified`.
- Cross-references: `configs/sub_f/semantic_vocab.yaml`, `configs/sub_f/unknown_family.yaml`, and `configs/sub_f/sentinel_inventory.yaml`.

## Section 10.5 telemetry

- Deterministic sampling seed: 20260523
- Candidate grid: direction_count in 8, 16, 24 crossed with magnitude_quantum_m in 0.25, 0.5, 1.0.
- Data read mode: `pq.ParquetFile(path).read()` per tile, not parent-directory reads.
- Geometry decode: `shapely.wkb.loads` from Sub-C little-endian WKB.
- Halt boundary: Halt 2 approved; `encoding_primitives.yaml` is `_status: LOCKED`; sentinel inventory BP2 is LOCKED; BP7 remains PLACEHOLDER.
