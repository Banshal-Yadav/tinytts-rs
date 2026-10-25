# Third-party notices

This crate embeds lexical data from the following sources. The runtime crate
(`tinytts-rs`) embeds none of them; they live only in `tinytts-g2p`.

## misaki 0.9.4 lexicon data

- Files: `us_gold.json`, `us_silver.json` (embedded here in compiled form as
  `src/lexicon.bin`)
- Source: the `misaki` Python package, <https://github.com/hexgrad/misaki>,
  PyPI `misaki==0.9.4`
- License: Apache License 2.0. The full license text ships with the misaki
  package and is available at <https://www.apache.org/licenses/LICENSE-2.0>.
- Provenance note: the gold and silver lexicons are distributed inside the
  misaki package as `misaki/data/us_gold.json` and `misaki/data/us_silver.json`.
  The misaki repository does not carry a separate provenance statement for
  these two files beyond the package-level Apache-2.0 license; they are
  treated as part of misaki itself. If a future misaki release states a
  different origin for these files, this notice must be updated before the
  embedded copy is regenerated.

## CMUdict

- File: `cmudict.dict` (embedded here in compiled form as `src/lexicon.bin`)
- Source: <https://github.com/cmusphinx/cmudict> (`cmudict.dict`)
- License (BSD-style, Carnegie Mellon University):

```
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer. The contents of this
   file are deemed to be source code.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. This work was
   supported in part by funding from the Defense Advanced Research Projects
   Agency, the Office of Naval Research and the National Science Foundation
   of the United States of America, and by member companies of the Carnegie
   Mellon Sphinx Speech Consortium. We acknowledge the contributions of many
   volunteers to the expansion and improvement of this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND ANY
EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY NOR ITS EMPLOYEES
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Regeneration

The embedded blob is regenerated with:

```
python g2p/gen_lexicon.py g2p/src/lexicon.bin
```

which reads the two misaki JSON files from the Python environment and
`cmudict.dict` from the spike folder. Any regeneration must re-check the
upstream sources for license changes before the new blob is committed.
