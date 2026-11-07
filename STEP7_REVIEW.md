# Step 7 review — verified saved outputs and transportation evidence

Step 7 is complete for its stated audit/export scope: 63 PASS, 2 WARN,
0 FAIL and 0 INCOMPLETE. It did not change PEBR or train new topic models.
The entire revised experimental programme is not yet complete.

## Independent inspection

- Recomputed NPMI, C_v and TD for all 60 exported baseline arrays using the
  exact original-text references. Recomputed the 12 five-seed aggregate rows
  and matched-topic stability. Maximum numerical difference across metric,
  aggregate and case-decomposition checks: 1.475e-15.
- Independently replayed four exported seed-0 native decoder samples. Maximum
  absolute difference from saved scores: 2.683e-7. The local audit's maximum
  over all 40 neural decoder replays was 4.769e-7; its maximum membership
  replay difference was 3.279e-6. No top-ten word set changed under CPU replay.
- Checked the recurrence counts against all 80 saved anchor/method/seed case
  records and visually inspected both recurrence figures. The ZIP does not
  include all final case-model factors, so the independent check does not
  claim to repeat every original-post membership inference here; the local
  Step 7 audit checked those factors before exporting its case records.
- Verified all 42 exported case-policy mean gains from the corresponding
  per-word probabilities and original target-token counts.
- Verified all four exported embedding arrays against their recorded aligned
  byte hashes. English coverage is 1,273/1,296 vocabulary words (98.23%);
  23 OOV words retain their fixed initialization. Chinese arrays use the
  recorded frozen multilingual/PCA adaptation, without English OOV substitution.
- All 11 frozen Step 7 runner files match their recorded hashes.

The full training checkpoints were inspected on the user's computer and were
not included in this compact ZIP. These checks support saved implementation
consistency; they do not prove that the published training configurations are
optimal for these corpora.

## What remains of the baseline concern

No vocabulary alignment, decoder extraction, native input or metric arithmetic
defect was demonstrated by this audit. All checked topic-score arrays are
nonconstant. The low neural coherence scores are real under the saved
configurations; they are not explained by a discovered top-word extraction bug.

Do not read kNNTM's first-20 versus last-20 loss increase as divergence:
its reconstruction target changes when augmentation starts after epoch 20.
For K12 seed0, the comparable augmented-phase means are:

| Corpus | Model | Epochs 21–40 mean loss | Epochs 181–200 mean loss |
| --- | --- | ---: | ---: |
| Transport-grievance | kNNTM | 203.3137 | 181.3884 |
| Chinese | kNNTM | 340.2219 | 299.4824 |
| Transport-grievance | GloCOM | 1289.6279 | 1043.9508 |
| Chinese | GloCOM | 5614.1076 | 4930.6858 |

Finite losses and a decreasing late-phase trend do not establish convergence
or adequate tuning. Retain the qualification “under the evaluated configurations.”
Genuine K10 BTM/kNNTM/GloCOM runs remain missing: 30 fits, both corpora and five
seeds. K8/K12 results remain valid; K16 must never be relabelled as K10 or K12.

## Transportation evidence: recommended interpretation

The best role is to explain actual evidence admission and transfer on original
transport complaints. An illustrative case does not prove semantic accuracy
for all posts or establish a cause of an incident.

First fresh protected bank 3712, K12; gains are nats per protected target token:

| Case/source row | Similarity sharing | Matched amount | Same-score prefix | PEBR |
| --- | ---: | ---: | ---: | ---: |
| Cab payment demand, 271 | 0.007057 | -0.056573 | -0.011183 | 0.017990 |
| Train delay, 357 | -0.062032 | -0.009788 | 0.000000 | 0.019671 |
| Coach facilities, 264 | 0.204559 | 0.115319 | 0.179390 | 0.127410 |
| Bus-service request, 14 | -0.017711 | 0.000000 | -0.005629 | 0.000000 |
| Unsuccessful example, 5 | -0.072247 | -0.020270 | -0.020270 | -0.030380 |

The cab example is a clear illustration: PEBR transfers from source row 277,
whose original excerpt concerns excess cab charges. The first bank increases
probabilities for genuine receiver words “fare” and “uber.” This anchor also
has positive mean protected gain in each of the four fresh outer banks.

The train example illustrates partial transfer: PEBR selected three donors,
but only source row 426 contributes at transfer. Rows 334/335 are unavailable;
their shares remain with the receiver. The realized mass is 1/9, not 1/3.
Do not present all three selected donors as transferred train-delay evidence.
The first-bank mean gain is positive although the word “delhi” loses probability.
The train anchor's other holdouts are mixed, and the prefix control has a larger
four-bank mean gain. Preserve these limits.

IMPORTANT CORRECTION: The bus example is NOT an explicit no-borrowing choice.
PEBR selected source row 95, which was unavailable at transfer. Offered mass
is 1/3; realized mass is zero. This illustrates receiver fallback. It must not
be described as PEBR rejecting all donor evidence.

A genuine explicit no-borrowing example exists at source row 508 in this bank:
its selected donor group is empty and offered mass is zero. It is the first
source-order eligible empty-action case with positive ordinary-sharing mass,
selected without consulting protected gain. PEBR's gain is zero; ordinary
sharing's gain is -0.262292. The prefix also chooses no borrowing, so this case
illustrates the admission option, not a unique advantage over the prefix.
Step 8 exports this case's unchanged/redacted original excerpt and all controls.

Retain the coach and unsuccessful cases in the supporting evidence. Do not
use automatically outcome-selected source row 26 as a representative success
rate or evidence that the rule always selects semantically matching donors.

## Recommended paper assets

Main paper: one concise evidence-decision case table, explaining useful
borrowing, explicit abstention or unavailable-donor fallback, and a limitation.
The fixed anchors and selection criteria must be disclosed. Include the strong
same-score prefix alongside ordinary and matched sharing when comparing outcomes.

Supplement: complete fixed-anchor holdouts, donor/source ledgers, unsuccessful
case, original five-seed recurrence figures and the existing protected audit.
Final-topic cases (selection mask3711) and protected prediction cases (outer3712)
use different action banks. They are complementary descriptions, not the same
before/after trace and not independent validation studies.

## Next implementation and closeout

Step 8: reference seeds1/2 and family-fold seeds3710/4710, studied separately
with the fixed revised rule, mask3711, K8/10/12 and five final seeds. This checks
sensitivity that repeated masks alone could not address.

After Step 8 review: complete the genuine K10 benchmarks, finish the baseline
budget review, regenerate retained corpus-size/parameter/runtime assets for
the revised rule, and produce a final main/supplement table-to-run ledger.
Only that closeout should mark the entire results package ready for rewriting
the manuscript. Further implementation should respond to demonstrated failures,
not continue searching for universally favorable rankings.
