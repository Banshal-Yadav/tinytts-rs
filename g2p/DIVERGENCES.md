# Intentional Divergences from Upstream Misaki

This document records the intentional design divergences where `tinytts-g2p`
and our reference (`proto3.py`) deliberately choose a different output from
upstream `misaki 0.9.4`. In all other cases, misaki is our oracle.

## 1. Time formats (H:MM)

- **Input example**: `9:30`, `6:45`, `12:00`
- **Upstream misaki**: Emits an unhandled placeholder character (`❓`, U+2753)
  for the colon token, resulting in e.g. `nˈIn❓ θˈɜɹTi`.
- **Our output**: `nˈIn: θˈɜɹTi` (the colon is retained as punctuation glued
  to the preceding hour digits, matching natural reading pause).
- **Reasoning**: Upstream misaki lacks a dedicated time rule in its rule-based
  cleaner for un-annotated text, so spaCy tokenization leaves the colon as
  an unhandled punctuation symbol. Emitting the colon as glued punctuation
  gives clean prosody in the acoustic model rather than an unhandled-character
  fallback.

## 2. Bare negative numbers

- **Input example**: `-7` (as a standalone token, e.g. "The temperature was -7 degrees")
- **Upstream misaki**: Drops the leading hyphen entirely or emits an empty
  token because the hyphen is treated as a dash/separator without preceding context.
- **Our output**: `mˈInəs sˈɛvən` ("minus seven").
- **Reasoning**: In English prose, a hyphen directly prefixed to digits represents
  a negative number and should be spoken as "minus". Dropping the minus sign
  causes severe semantic error (e.g. speaking "-7 degrees" as "7 degrees").

## 3. OOV Fallback Policy

- **Input example**: Novel proper nouns or rare jargon not in the lexicon.
- **Upstream misaki**: Delegates OOV words to `EspeakFallback`, which links to
  the external C library `espeak-ng`.
- **Our output**: A self-contained, rule-based English letter-to-sound fallback
  combined with compound splitting, accompanied by a stderr warning log.
  Words are never dropped from the synthesized audio stream.
- **Reasoning**: `tinytts-g2p` must remain zero-dependency and static-embeddable
  in pure Rust without C compiler toolchains or external system libraries.
  In TTS synthesis, speaking an approximate pronunciation is strictly preferable
  to dropping the word completely (which produces silent gaps in the audio).

## 4. Heteronym Disambiguation (POS Tagger)

- **Input example**: Heteronyms such as `lead`, `read`, `bass`, `wind`, `live`.
- **Upstream misaki**: Runs a full spaCy POS tagger (`en_core_web_sm`, ~15 MB
  neural pipeline) to determine whether a token is noun, verb, or adjective.
- **Our output**: Resolves heteronyms to their DEFAULT (canonical / noun) form,
  augmented with zero-cost syntactic context rules (e.g. `used to` -> `jˈust`,
  sentence-final `there` -> `ðˈɛɹ`, sentence-final `in` -> `ˈɪn`,
  sentence-initial `That` -> `ðˈæt`).
- **Reasoning**: Shipping Python spaCy and neural POS models would violate the
  core goal of a zero-dependency, lightweight engine. Documented limitation.
