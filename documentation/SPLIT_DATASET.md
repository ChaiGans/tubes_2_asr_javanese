# Your Data Statistics:
Total speakers: 70
Native (n): 14 speakers
Non-native (nn): 56 speakers

Total utterances: 2,100
Native: 420 utterances
Non-native: 1,680 utterances

# Recommended Split:
## Test Set (20% of speakers, speaker-disjoint):
Native: ~3 speakers (21% of 14 = 2.94 ≈ 3)
Non-native: ~11 speakers (20% of 56 = 11.2 ≈ 11)
Total: ~14 speakers for test

## Train+Val Set (80% of speakers):
Native: ~11 speakers
Non-native: ~45 speakers
Total: ~56 speakers

Then split their utterances:
70% utterances → Training
10% utterances → Validation