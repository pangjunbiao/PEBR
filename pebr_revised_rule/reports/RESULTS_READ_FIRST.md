# Focused PEBR rule revision: actual-corpus development results

The export succeeded. All 389 numerical files were verified. The 28 baseline fits reproduced with identical descriptor lists.
The study computed 510 new development fits: two corpora, three topic counts, seventeen candidate/control policies, five final seeds.
These are conditional on one selection bank (mask base 2711), not independent mask/corpus replications. All declared settings are retained.

## Recommended candidate to validate next

Receiver-preserving predictive selection with the existing forward search and a fixed positive offered cap of 1/3.
The score rewards token-weighted hidden-word benefit and charges token-weighted visible-word prediction loss.
The additional donor-swap search is not adopted. Searching offered borrowing weights is not retained in this candidate.
Reference fitting, retrieval, fold/family exclusions, transfer fallback and final NMF remain unchanged.

## Comparison with original PEBR

| Collection | K | Original NPMI | Revised NPMI | Original C_v | Revised C_v | Original stability | Revised stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| transport_grievance | 8 | 0.0465 | 0.0777 | 0.4853 | 0.5116 | 0.7250 | 0.7704 |
| transport_grievance | 10 | 0.0490 | 0.0881 | 0.5083 | 0.5508 | 0.7833 | 0.8053 |
| transport_grievance | 12 | 0.0470 | 0.0603 | 0.5240 | 0.5346 | 0.6724 | 0.6550 |
| chinese_pooled | 8 | 0.2168 | 0.2399 | 0.6536 | 0.6717 | 0.6675 | 0.7620 |
| chinese_pooled | 10 | 0.2288 | 0.2271 | 0.6740 | 0.6746 | 0.7673 | 0.7572 |
| chinese_pooled | 12 | 0.2415 | 0.2387 | 0.6917 | 0.6891 | 0.8368 | 0.7087 |

## Evidence against simpler controls

Higher below means higher mean NPMI AND C_v. These counts do not establish statistical significance.

| Comparator | Higher in both coherence measures |
|---|---:|
| NMF without borrowing | 6/6 |
| Ordinary similarity sharing | 5/6 |
| Original rule matched-amount control (different quantity reference) | 6/6 |
| Full-pool LOO | 4/6 |
| Original PEBR | 4/6 |
| Original predictively scored similarity prefixes | 6/6 |
| Receiver-preserving forward rule with borrowing-weight search | 4/6 |
| NEW matched-amount control for the revised candidate | 6/6 |
| Same token weighting, without visible-word loss charge | 5/6 |
| Revised candidate without downside penalty | 5/6 |
| Revised candidate with individual-donor scoring | 5/6 |
| Revised candidate with single-view selection | 6/6 |
| Similarity-ranked prefixes under the SAME revised score and fixed cap | 6/6 |

## Limits and decision

This is a useful development improvement, not a completed confirmation study. New matched controls match each receiver’s realized mass and contributing donor count; all checks passed.
The revised candidate improves both mean coherence measures over original PEBR in four settings. Chinese K=10 is mixed and Chinese K=12 is lower on both measures.
Chinese K=12 stability also decreases substantially. The candidate is not uniformly superior, and higher coherence does not establish better service retrieval or analyst performance.
Independent masking/selection repeats and a protected-word evaluation of the frozen revised rule remain necessary before making reliability claims.
Do not replace manuscript numbers selectively or combine revised-method results with old-method component, sensitivity or corpus-size figures. A method change requires version-consistent reporting.
The practical next step is to validate this simpler revised candidate. The old unchanged Step 4 repetition package should remain paused.

All numerical summaries, paired seed effects, borrowing checks and unfavorable comparisons are included in the accompanying CSV/JSON files.