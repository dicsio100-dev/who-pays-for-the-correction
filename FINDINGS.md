# Findings log — "Who pays for the correction?"

Lab notebook. Every number comes from a script in `code/`, run on the data in `~/veritas-data` (or `$VERITAS_DATA`)
(not in the repo). The explanation page reads the same numbers from `docs/data.json` (`export_site.py`).
Frozen on 2026-11-14 after switching to two-sided p-values and one fixed random stream per statistic
(`analyze.seeded`), so re-running gives identical numbers.

## Setup (fixed before looking at results)

- Data: Mozilla Common Voice French 27.0 (validated clips), accent = speaker's self-declared variant:
  France (`Français de métropole`), Canada (`Français d'Amérique du Nord`),
  Sub-Saharan Africa (`Français d'Afrique subsaharienne et des îles africaines`).
  50 speakers per group, 4 clips per speaker (seed 14). Complement: OpenSLR 57 Yaoundé read speech (proverbs), 30 speakers, recorded in 2003 by the U.S. Military
  Academy's CTELL, Apache 2.0.
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
- p-values switched from one-sided to two-sided (stricter): this made every result weaker.
- The other two choices cut both ways. Using all speakers for word error rate made the Canada clean gap stronger
  (p = 0.068 on the test half, 0.006 on all speakers). Cross-fitting made the clean repeat gaps weaker
  (Africa 0.018 → 0.126, Yaoundé 0.040 → 0.084) and the phone ones a little stronger (0.062 → 0.052,
  0.014 → 0.008). Both versions are reported below.
- Added after the first look and reported whatever came out: key words only (Result 6), the repeat rule among
  correctly heard sentences and the error/score split (end of Result 3).
- Added on Sunday, after a second outside review: the three-way split and mean scores (end of Result 3), the
  sentence-length check for fix 2 (Result 5), and four more fixes (Result 7). All post-hoc.

## Data integrity (checks.py, analyze.py)

- 720 / 720 clips transcribed in each condition, 0 duplicates, 0 empty outputs, 0 missing confidences.
- Hallucinations (large-v3): 1 in 1440 transcriptions (Whisper wrote "Réalisé par Neo035", Africa, tune half, clean). Whisper small: at
  least one long loop ("Merci d'avoir regardé cette vidéo…", 138 errors in one Canadian clip).
- Recording quality (rough SNR, digital silence excluded): France 50.5 dB, Canada 49.3, Africa 49.3, Yaoundé 43.5.
- Sentence length (normalised tokens): France 9.4, Africa 10.2.
- Common Voice 27.0 French validated clips: 792,011 in total; Sub-Saharan Africa 1,846 (0.23 %, 84 speakers),
  France 505,272 (63.8 %), Canada 12,893 (1.6 %), no variant 244,265 (30.8 %). Counted from `validated.tsv`.

## Result 1 — Whisper makes 1.5 to 1.8 times as many errors for Canadian and Sub-Saharan African French (solid)

Ratios to France, all speakers: clean Canada 1.70, Africa 1.47; phone Canada 1.72, Africa 1.84.

Word error rate, all speakers:

| Group | Clean | Phone | Gap vs France, clean | Gap vs France, phone |
|---|---|---|---|---|
| France (50) | 7.3 % | 7.9 % | — | — |
| Canada (50) | 12.3 % | 13.6 % | +5.1 pts, p = 0.006 | +5.7 pts, p = 0.010 |
| Sub-Saharan Africa (50) | 10.7 % | 14.6 % | +3.4 pts, p = 0.028 | +6.7 pts, p = 0.002 |
| Yaoundé (30, other corpus) | 17.9 % | 18.8 % | +10.6 pts, p < 0.002 | +10.8 pts, p < 0.002 |

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

4 of 8 below 0.05. Canada is not asked to repeat more (−6.5 to +7 per 100, all p > 0.19). Slip-through gaps: none
significant except Yaoundé having fewer (it is asked to repeat more). Do NOT claim Canadians get more silent errors.

Why the rule costs so much: the sentence score is a weak error detector. AUC (chance a correct clip outscores a
misheard one), phone, all speakers: France 0.62, Canada 0.66, Africa 0.72, Yaoundé 0.80. Not worse for Canada
(no "overconfident on Canadian French" effect).

Is the rule unfair at equal error, or does it only follow Whisper's extra mistakes? Share of sentences Whisper heard
with NO error that the rule still asks to repeat (`false_alarm` in analyze.py), gap vs France, two-sided p:

| | test, clean | test, phone | cross-fit, clean | cross-fit, phone |
|---|---|---|---|---|
| France (reference) | 63 % | 62 % | 74 % | 72 % |
| Sub-Saharan Africa | +13.7, p = 0.196 | +9.1, p = 0.364 | +4.0, p = 0.564 | +7.9, p = 0.268 |
| Yaoundé | +12.1, p = 0.278 | +11.4, p = 0.25 | +2.1, p = 0.726 | +3.6, p = 0.598 |
| Canada | +5.4, p = 0.582 | −9.1, p = 0.37 | −6.5, p = 0.29 | −10.1, p = 0.108 |

All 8 African gaps positive (+2 to +14 per 100), none significant (38–102 correct sentences per group).

Rough split of each repeat gap into three parts (point estimates, no range): (a) more misheard sentences (the African
group's own share, France's repeat rates), (b) misheard sentences flagged more often, (c) correctly heard sentences
asked more often. Phone cross-fit, Sub-Saharan Africa: 10.0 = 2.2 + 4.4 + 3.4. Across the 8 versions, (a) + (b), the
part that comes with Whisper's mistakes, is 56 to 89 % of the gap (47 to 84 % when the split uses France's misheard
share for (b) and (c) instead); (c) is 0.9 to 6.5 points.
Why (b): African misheard sentences score lower (phone, all speakers: −0.30 against −0.24), so the line catches more
of them (92 % against 84 %; the flip side is that 16 % of France's misheard sentences slip through, against 8 %).
They also carry more wrong words (2.6 against 1.9), but that is not the reason: Canadian misheard sentences carry
even more (2.9) and are caught less (80 %). On correctly heard sentences the average score on the phone line is the
SAME (−0.185 against −0.188; on normal audio, test half, Africa's is lower, −0.198 against −0.169): (c) is not a
lower average; it comes from fewer very confident correct sentences above the line, and it is not significant.

Correction (Sunday): Saturday's two-way split put (b) and (c) together as "lower confidence scores" and read it as
"only 1 to 4 points come from extra mistakes". That over-read the split: (b) is misheard sentences, so those repeats
come with real mistakes; they are caught more because they score lower, not because the mistakes are bigger.

Reading: most of the extra repeats come with Whisper's mistakes (more of them, and more of them caught): the rule
turns the recognizer's accent gap into callers' time. Among correctly heard sentences the gap points the same way in all 8
versions but is not significant in any: a hint, not a proof.

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

All savings p < 0.002 (no resample crossed zero). Gap with France in words re-said per call:

| Group | Whole sentence | Targeted | Change (does targeted close it?) |
|---|---|---|---|
| Canada | +1.24 [−0.15, +2.77] p = 0.094 | +1.84 [+0.84, +2.87] p = 0.002 | −0.60 [−1.45, +0.33] p = 0.23 |
| Sub-Saharan Africa | +2.25 [+0.82, +3.74] p = 0.002 | +2.21 [+1.24, +3.22] p < 0.002 | +0.04 [−0.99, +1.17] p = 0.91 |
| Yaoundé | +2.90 [+1.47, +4.37] p < 0.002 | +2.19 [+1.17, +3.19] p < 0.002 | +0.70 [−0.23, +1.80] p = 0.15 |

Sentence length (added Sunday): sentences are not the same length in every group (test half: France 9.3 words,
Sub-Saharan Africa 10.2, Canada 9.7, Yaoundé 9.3), and a longer sentence costs more words to repeat. Per word spoken,
shown in words at France's sentence length, the Sub-Saharan African gap is +1.60 (p = 0.014) with whole-sentence
repeats and +1.74 (p < 0.002) with targeted confirmation; Canada +0.98 (p = 0.126) and +1.64 (p < 0.002); Yaoundé
+2.98 and +2.25 (both p < 0.002). About half a word of the African gap above is length; the conclusion is unchanged.

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
- The check that could have overturned Result 1 (extra African/Canadian errors caused by people misreading the
  sentence, not by Whisper) is NOT met: 0 misreadings in the Canadian and African Common Voice clips checked. Artifacts appear in every group: noise, not the gap.

## Result 6 — Key words only: names and numbers fail for everyone (key_info.py, added after the first look)

Key words = a capitalised word that does not start the sentence (a name) or a word with a digit. Clips with at least
one key word only (66–75 per group, Yaoundé has none), all speakers:

| Group | Key words wrong, clean | Gap vs France | Key words wrong, phone | Gap vs France |
|---|---|---|---|---|
| France | 31 % | — | 35 % | — |
| Canada | 29 % | −2.2 pts, p = 0.76 | 37 % | +2.2 pts, p = 0.79 |
| Sub-Saharan Africa | 39 % | +7.7 pts, p = 0.22 | 43 % | +8.7 pts, p = 0.14 |

Re-running the repeat rule with "a key word is wrong" as the error to catch (cross-fit): no significant gap either
(Africa +1.5 clean p = 0.86, +9.1 phone p = 0.15; Canada −8.8 p = 0.21, −13.6 p = 0.068). Reading: Whisper gets
about a third of names and numbers wrong whoever speaks; the group gap is not resolvable on 66–75 clips. A phone
agent should confirm names for every caller.

## Result 7 — Four more fixes tried on Sunday (fixes.py, post-hoc, phone line only, no new audio)

Same rule set-up: accent-blind, settings and line fitted on tune speakers (all groups pooled) for slip <= 5 %, judged
on the other half and cross-fitted. Words re-said assume, as before, that what is said again is heard right.

- **Read the sentence back** ("I heard X, is that right?") instead of asking for a repeat: the caller says one word
  if it was right, one word plus the sentence if it was wrong. Same calls checked, so same safety. Words re-said per
  call, all speakers: France 6.74 → 3.80 (−44 %), Canada 6.27 → 4.31 (−31 %), Sub-Saharan Africa 8.44 → 6.17 (−27 %),
  Yaoundé 7.67 → 5.92 (−23 %). The Sub-Saharan African gap with France does not shrink (+1.70 → +2.37 words; change
  not significant, p = 0.24): once false alarms cost one word, what is left is the error-rate gap. It also widens
  the gap for Canada (−0.47 → +0.51 words, change p = 0.036) and Yaoundé (+0.94 → +2.12, p = 0.038) on all speakers,
  not on the test half (p ≥ 0.24): it helps France most. Optimistic: it
  assumes the caller notices every wrong read-back (a homophone read back sounds right), and it does not count the
  time spent listening to the read-back.
- **A line that adapts to each caller** (the score minus the mean score of the same caller's other sentences, an
  accent-blind way to correct a caller-level shift): tuning picks no adaptation. Forced, it evens checks out only by
  asking France and Canada callers more (France 77 % → 86 % at half strength, 92 % at full; Canada 70 → 80 → 88 %;
  Africa 87 → 89 → 88 %; Yaoundé 89 → 83 → 83 %). It also uses the
  caller's later sentences, so it is optimistic for a caller's first sentence.
- **Two recognisers** (large-v3 and turbo): a bonus when both write the same text, or a small logistic-regression
  alarm on accent-blind features (both scores, agreement, weakest word, length, no-speech probability): neither
  shrinks the checks gap (Africa +8 and +11 points with them, against +10). Choosing between the two transcripts
  (keep turbo's when it is clearly surer) cuts word errors for every group on the test half (France 6.9 → 5.5 %,
  Canada 13.7 → 13.3 %, Africa 15.9 → 14.4 %, Yaoundé 12.8 → 9.9 %; no range) without narrowing the gaps with
  France (Canada 6.8 → 7.8 points, Africa 9.0 → 8.9, Yaoundé 5.9 → 4.4).
- **Same line, same safety?** Under one line, Yaoundé callers are asked the most yet have the fewest mistakes slip
  through (0.8 % against 6.0 % for France, all speakers, gap −5.2 points [−9.2, −1.3]); Canada 9.5 % (p = 0.22),
  Africa 4.5 % (p = 0.63). Yaoundé is a different corpus, so this may be the recording set-up, not the accent.
  A separate line per group (label-aware, so not usable by an agent that does not know the caller's accent) evens
  checks out on paper (81 / 84 / 84 / 80 %), but it lets more Yaoundé mistakes through (6.7 %), and its gaps
  (Africa +3.5 points [−6, +13], p = 0.57) are too uncertain with 25 speakers per half to show the gap closes. Not a
  finding.

Reading: none of the agent-side fixes closes the gap with France. Reading back is the cheapest change for every
caller. Fixing the gap itself points back to the recognizer.

## Limits to state on the page

- 50 speakers per group (25 in each half): error-rate gaps of about 3 points are resolvable (3.4, p = 0.028), repeat
  gaps need about 10 points (10, p = 0.052).
- The repeat-rule ranges keep the line fixed; re-choosing it in every resample would widen them a little.
- Whisper's training data is not public, so it may include some Common Voice or OpenSLR recordings.
- Accent = self-declared variant; "Africa" pools many countries. Common Voice is read Wikipedia text, not calls.
- Simulated phone line, not real calls. The agent rule assumes a repeated sentence is then heard correctly.
- Many comparisons are reported; a single p near 0.05 is a hint, not a proof. Result 7 alone adds more than a hundred
  post-hoc comparisons; a few p below 0.05 there are expected by chance.

## Sources checked

- ISQ, Immigrants selon le pays de naissance (2024p): Cameroon 9,127 = 15.4 %, first.
- OIF, La langue française dans le monde 2022, synthesis p. 7: daily French speakers 47.4 % Sub-Saharan Africa
  and Indian Ocean, 14.6 % Maghreb–Middle East, 31.2 % Europe, 6.6 % Americas, 0.3 % Asia-Oceania.
- Common Voice terms (Mozilla Data Collective): CC0 but "forbidden to re-host or re-share this dataset" and to
  identify speakers → no Common Voice audio is published.
