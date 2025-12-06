# Your Data Statistics:
Total speakers: 70
Native (n): 14 speakers
Non-native (nn): 56 speakers

Total utterances: 2,100
Native: 420 utterances
Non-native: 1,680 utterances

# Split Strategy (Hybrid):

## Test Set (20% of speakers, SPEAKER-DISJOINT):
- **7 native speakers** (50% of 14 native speakers)
- **7 non-native speakers** (12.5% of 56 non-native speakers)
- **Total: 14 test speakers** (20% of 70 total speakers)
- Speakers in the test set will NOT appear in train or val.

## Train+Val Set (80% of speakers, UTTERANCE-LEVEL SPLIT):
- Remaining speakers: 7 native + 49 non-native = 56 speakers
- All utterances from these 56 speakers are pooled together
- Then randomly split per-utterance (not per-speaker):
  - **87.5% of remaining utterances → Training**
  - **12.5% of remaining utterances → Validation**
- NOTE: Same speaker's utterances may appear in BOTH train and val sets.