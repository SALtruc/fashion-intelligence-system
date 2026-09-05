Quarantined batch-2 candidate
=============================

910405.jpg - flagged by verify_external_data.py as matching the provided TRAIN set
(dHash distance 2). Investigated and found to be a hash collision, NOT a leak:
  - mean absolute pixel difference to its nearest 'match' was 124.87/255.
    A true duplicate sits near 0.
  - visual check: the crop is a dark rectangular cosmetic on a grey surface;
    its four nearest 'matches' are women wearing dresses and tops.

Excluded anyway so the shipped set carries zero gate flags. Kept here, not deleted,
so the judgement can be re-checked.
