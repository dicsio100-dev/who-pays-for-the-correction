# Findings log — "Who pays for the correction?"

Lab notebook. Every number comes from a script in `code/`, run on the data in `~/veritas-data` (or `$VERITAS_DATA`)
(not in the repo). The explanation page reads the same numbers from `site/data.json` (`export_site.py`).
Frozen on 2026-11-14 after switching to two-sided p-values and one fixed random stream per statistic
(`analyze.seeded`), so re-running gives identical numbers.

## Setup (fixed before looking at results)

- Data: Mozilla Common Voice French 27.0 (validated clips), accent = speaker's self-declared variant:
  France (`Français de métropole`), Canada (`Français d'Amérique du Nord`),
  Sub-Saharan Africa (`Français d'Afrique subsaharienne et des îles africaines`).
  50 speakers per group, 4 clips per speaker (seed 14). Complement: OpenSLR 57 Yaoundé read speech, 30 speakers.
- Speakers (not clips) split 50/50 into tune / test.
- Phone condition: 300–3400 Hz band-pass, 8 kHz, G.711 mu-law, back to 16 kHz.
- Model for error rates: whisper-large-v3 via Groq, language=fr, temperature 0.
- Text normalisation (both sides): lowercase, numbers to digits, accents and punctuation removed (`common.normalise`).
- Misheard clip = at least one word error. Agent rule = ask to repeat if sentence confidence (avg log-prob) < t,
  t = loosest threshold keeping errors that slip through <= 5 % on tune speakers, all groups pooled.
- Uncertainty: 1000 bootstrap resamples of speakers. p = two-sided bootstrap p (twice the smaller tail around 0).

### Protocol choices made after the first look (disclosed)
- Word error rate uses **all** speakers: nothing is tuned for it, so the tune/test split is not needed.
- The repeat rule is reported both on the pre-declared test half and **cross-fitted** (halves swap roles),
  so every speaker is judged once with a line tuned on other people.
- p-values switched from one-sided to two-sided (stricter). Both changes made results weaker, not stronger.

## Data integrity (checks.py, analyze.py)

- 720 / 720 clips transcribed in each condition, 0 duplicates, 0 empty outputs, 0 missing confidences.
- Hallucinations (large-v3): 1 in 1440 transcriptions ("réalisé par neo035", Africa, clean). Whisper small: at
  least one long loop ("Merci d'avoir regardé cette vidéo…", 138 errors in one Canadian clip).
- Recording quality (rough SNR, digital silence excluded): France 50.5 dB, Canada 49.3, Africa 49.3, Yaoundé 43.5.
- Sentence length (normalised tokens): France 9.4, Africa 10.2.
- Common Voice 27.0 French validated clips: 792,011 in total; Sub-Saharan Africa 1,846 (0.23 %, 84 speakers),
  France 505,272 (63.8 %), Canada 12,893 (1.6 %), no variant 244,265 (30.8 %). Counted from `validated.tsv`.

## Result 1 — Whisper makes about twice as many errors for Canadian and African French (solid)

Word error rate, all speakers:

| Group | Clean | Phone | Gap vs France, clean | Gap vs France, phone |
|---|---|---|---|---|
| France (50) | 7.3 % | 7.9 % | — | — |
| Canada (50) | 12.3 % | 13.6 % | +5.1 pts, p = 0.006 | +5.7 pts, p = 0.010 |
| Sub-Saharan Africa (50) | 10.7 % | 14.6 % | +3.4 pts, p = 0.028 | +6.7 pts, p = 0.002 |
| Yaoundé (30, other corpus) | 17.9 % | 18.8 % | +10.6 pts, p < 0.001 | +10.8 pts, p < 0.001 |

Robust to: removing sentences with a proper noun (clean: 5.2 / 11.1 / 9.1 %), keeping only above-median SNR clips
(clean: 7.4 / 10.7 / 9.4 %), standardising sentence length to France's mix (Africa misheard 49 % → 48.1 %).
Test half only (25 speakers): Canada clean gap p = 0.068 (not significant there), all other gaps p <= 0.042.

Same direction as Nieto, Angelika-Nikita, Sarkis & Yang, "Dialect Bias in Speech Recognition Across 10 Spanish
and French Varieties", Interspeech 2026, doi:10.21437/Interspeech.2026-458, Table 2 (podcasts, Whisper v3):
male speakers France 4.90 %, Canada 6.24 %, Senegal 8.82 %, Ivory Coast 8.63 %; female speakers 3.67 / 6.63 /
5.00 / 4.29 %. They localise the bias in the decoder (fine-tuning experiment, section 5.3).
PDF: https://www.isca-archive.org/interspeech_2026/nieto26_interspeech.pdf

## Result 2 — The phone line widens the gap for Sub-Saharan African speakers only (solid)

Difference-in-differences (phone gap minus clean gap, vs France), all speakers:
Africa +3.2 pts [+1.2, +5.3], p = 0.002. Canada +0.6 pts, p = 0.43. Yaoundé +0.2 pts, p = 0.76.
→ Not a general "accent + phone" effect; it holds for the Common Voice African group, not for Yaoundé.

## Result 3 — The same repeat rule asks African-accented callers more often (consistent, moderate evidence)

At <= 5 % slipped errors, repeat share on test speakers: clean 67 / 74 / 83 / 82 % (France / Canada / Africa /
Yaoundé), phone 68 / 62 / 83 / 85 %.
Repeat gap vs France, 8 comparisons (2 African groups × clean/phone × test half / cross-fit), all positive
(+8 to +17 per 100 calls); two-sided p:

| | test, clean | test, phone | cross-fit, clean | cross-fit, phone |
|---|---|---|---|---|
| Sub-Saharan Africa | 0.018 | 0.062 | 0.126 | 0.052 |
| Yaoundé | 0.040 | 0.014 | 0.084 | 0.008 |

4 of 8 below 0.05. Canada is not asked to repeat more (−6 to +7 per 100, all p > 0.19). Slip-through gaps: none
significant except Yaoundé having fewer (it is asked to repeat more). Do NOT claim Canadians get more silent errors.

Why the rule costs so much: the sentence score is a weak error detector. AUC (chance a correct clip outscores a
misheard one), phone, all speakers: France 0.62, Canada 0.66, Africa 0.72, Yaoundé 0.80. Not worse for Canada
(no "overconfident on Canadian French" effect).

## Result 4 — Fix 1: telling Whisper the language narrows the gap (language_hint.py)

Phone, test speakers, whisper-large-v3 without vs with `language=fr`:

| Group | Wrong language guessed | WER guessing | WER told French | Gap with France narrowed by |
|---|---|---|---|---|
| France | 2 / 100 (English, Portuguese) | 6.8 % | 6.9 % | — |
| Canada | 7 / 100 (English 6, Georgian 1) | 16.5 % | 13.7 % | 2.9 pts [+0.2, +6.7], p = 0.012 |
| Sub-Saharan Africa | 9 / 100 (English 5, Korean 2, Italian, Haitian) | 17.1 % | 15.9 % | 1.3 pts [0.0, +3.3], p = 0.058 |

Several "English" outputs are translations of the French speech. All other results already use `language=fr`.

## Result 5 — Fix 2: confirm only the doubtful words lowers the bill for everyone, not the gap (policy_words.py)

Whisper small (local, per-word confidence) was too weak on phone audio (WER 17.7 / 43.8 / 33.2 %): reaching 5 % slip
meant flagging nearly every word. So this uses large-v3-turbo (local, int8, greedy, per-word confidence), phone line,
WER 8.4 / 18.0 / 17.8 / 16.4 % (France / Canada / Sub-Saharan Africa / Yaoundé). Both rules are tuned on the TUNE
half (all groups pooled) for slip <= 5 %, judged on TEST speakers. Same optimistic assumption for both: what the
caller re-says is heard right. Words re-said = whole sentence for a repeat, flagged words for targeted.

| Group | Whole sentence: words/call | Targeted: words/call | Saved [95 %] | Interrupted (sentence → targeted) | Slip (s / t) |
|---|---|---|---|---|---|
| France | 5.67 | 3.44 | 2.23 [1.58, 2.96] (39 %) | 63 % → 96 % | 8 % / 7 % |
| Canada | 6.91 | 5.28 | 1.63 [1.05, 2.26] (24 %) | 73 % → 98 % | 5 % / 7 % |
| Sub-Saharan Africa | 7.92 | 5.65 | 2.27 [1.51, 3.13] (29 %) | 78 % → 100 % | 3 % / 3 % |
| Yaoundé | 8.57 | 5.63 | 2.93 [2.17, 3.73] (34 %) | 93 % → 100 % | 2 % / 3 % |

All savings p < 0.001. Gap with France in words re-said per call:

| Group | Whole sentence | Targeted | Change (does targeted close it?) |
|---|---|---|---|
| Canada | +1.24 [−0.15, +2.77] p = 0.094 | +1.84 [+0.84, +2.87] p = 0.002 | −0.60 [−1.45, +0.33] p = 0.23 |
| Sub-Saharan Africa | +2.25 [+0.82, +3.74] p = 0.002 | +2.21 [+1.24, +3.22] p < 0.001 | +0.04 [−0.99, +1.17] p = 0.91 |
| Yaoundé | +2.90 [+1.47, +4.37] p < 0.001 | +2.19 [+1.17, +3.19] p < 0.001 | +0.70 [−0.23, +1.80] p = 0.15 |

Reading: targeted confirmation cuts the words callers say again by a quarter to 40 % in every group, but the gap
with France stays the same size (no change is significant). It also turns most calls into at least one question
(96–100 %): it asks for less, but asks almost everyone. The fix lowers the bill; it does not make it fair. This
points back to the recognizer itself (more African-accented French in training data), not the agent's rule.
Limit: one model (turbo, greedy, CPU), test half only (25 speakers per group), no cross-fit for this comparison.

## Human listening check (Haris, native French speaker, 2026-11-14)

20 clips, 5 per group (`make_check_page.py`, answers in `results/human_check.json`).
- Reference text matches the speech: 18 / 20. France 4/5 (dataset typo "protugais": speaker and Whisper said
  "portugais"), Canada 5/5, Sub-Saharan Africa 5/5, Yaoundé 4/5 (speaker said "humanité" instead of "amitié").
- Clips with counted errors: 12. All counted errors real: 8; some: 2 (Africa, e.g. inaudible agreement
  "consommés/consommées"); none: 2 (Canada "l'article L." truncated sentence; the Yaoundé misreading).
- The contradictor's "flip" condition (extra African/Canadian errors caused by misreading) is NOT met: 0 misreadings
  in the Canadian and African Common Voice clips checked. Artifacts appear in every group: noise, not the gap.

## Limits to state on the page

- 50 speakers per group (25 in each half): intervals are wide; gaps under ~5 points are not resolvable.
- Accent = self-declared variant; "Africa" pools many countries. Common Voice is read Wikipedia text, not calls.
- Simulated phone line, not real calls. The agent rule assumes a repeated sentence is then heard correctly.
- Many comparisons are reported; a single p near 0.05 is a hint, not a proof.

## Sources checked

- ISQ, Immigrants selon le pays de naissance (2024p): Cameroon 9,127 = 15.4 %, first.
- OIF, La langue française dans le monde 2022, synthesis p. 7: daily French speakers 47.4 % Sub-Saharan Africa
  and Indian Ocean, 14.6 % Maghreb–Middle East, 31.2 % Europe, 6.6 % Americas, 0.3 % Asia-Oceania.
- Common Voice terms (Mozilla Data Collective): CC0 but "forbidden to re-host or re-share this dataset" and to
  identify speakers → no Common Voice audio is published.
