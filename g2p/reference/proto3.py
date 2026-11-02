"""HYBRID PROTOTYPE v3: gold+silver lexicon, CMU fallback, context rules.

Fixes over proto2:
  - punctuation glued to previous word (no space before , . ! ? ; :)
  - hyphenated compounds joined, later primaries demoted to ˌ
  - 'a' article -> ɐ, 'the' -> ði before vowel else ðə
  - 'to' -> tʊ before vowel else tə
  - sentence-initial capitalized That -> ðˈæt
  - capitalized non-special words: ˌ before first vowel if no stress mark
  - CMU ER rules: ER1/ER2 -> ɜɹ, ER0 -> əɹ medial / ɹ final
  - BOM strip
"""
import json, os, re, time
from collections import Counter
import misaki

HERE = os.path.dirname(os.path.abspath(__file__))
d = os.path.join(os.path.dirname(misaki.__file__), 'data')
GOLD = json.load(open(os.path.join(d, 'us_gold.json'), encoding='utf-8'))
SILVER = json.load(open(os.path.join(d, 'us_silver.json'), encoding='utf-8'))

def grow(dd):
    e = {}
    for k, v in dd.items():
        if len(k) < 2:
            continue
        if k == k.lower() and k != k.capitalize():
            e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **dd}

GOLD = grow(GOLD)
SILVER = grow(SILVER)

CMU = {}
for line in open(os.path.join(HERE, 'cmudict.dict'), encoding='utf-8', errors='replace'):
    line = line.strip()
    if not line or line.startswith(';;;'):
        continue
    parts = line.split()
    CMU.setdefault(parts[0].lower(), tuple(parts[1:]))

MAP = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'W', 'AY': 'I',
    'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɜ',
    'EY': 'A', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
    'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'OW': 'O', 'OY': 'Y', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}
UNSTRESSED = {
    'AH': 'ə', 'IH': 'ɪ', 'IY': 'i', 'EY': 'A', 'OW': 'O', 'UW': 'u',
    'AE': 'æ', 'AA': 'ɑ', 'AO': 'ɔ', 'EH': 'ɛ', 'UH': 'ʊ', 'AW': 'W',
    'AY': 'I',
}
VOWELS = set('AIOWYaiuæɑɒɔəɛɜɪʊʌᵻ')

def cmu_convert(phones):
    out = []
    n = len(phones)
    for i, ph in enumerate(phones):
        base = ph.rstrip('012')
        stress = ph[len(base):]
        if base == 'ER':
            if stress in ('1', '2'):
                sym = 'ɜɹ'
            else:
                # ER0: ɹ word-final, əɹ otherwise
                sym = 'ɹ' if i == n - 1 else 'əɹ'
        else:
            sym = MAP.get(base)
            if sym is None:
                return None
            if stress == '0':
                if base == 'IH':
                    # misaki reduces unstressed IH to ə in CMU fallback words
                    sym = 'ə'
                elif base == 'AH':
                    # AH0 -> ᵊ before nasal+consonant (applicants, questions)
                    if i + 2 < n and phones[i + 1].rstrip('012') in ('N', 'M', 'NG') \
                            and phones[i + 2].rstrip('012') not in ('AA', 'AE', 'AH', 'AO', 'AW',
                                                                    'AY', 'EH', 'ER', 'EY', 'IH',
                                                                    'IY', 'OW', 'OY', 'UH', 'UW'):
                        sym = 'ᵊ'
                    else:
                        sym = 'ə'
                else:
                    sym = UNSTRESSED.get(base, sym)
        if stress == '1':
            out.append('ˈ')
        elif stress == '2':
            out.append('ˌ')
        out.append(sym)
    return ''.join(out)

def flap(ps):
    out = []
    for i, c in enumerate(ps):
        if c == 't':
            prev_v = i > 0 and ps[i - 1] in VOWELS
            next_v = i + 1 < len(ps) and (ps[i + 1] in VOWELS or
                                          (ps[i + 1] == 'ɹ' and i + 2 < len(ps) and ps[i + 2] in VOWELS))
            if prev_v and next_v:
                out.append('T')
                continue
        out.append(c)
    return ''.join(out)

def flap_all(ps):
    """misaki's final pass: ALL intervocalic ɾ -> T (paradee version)."""
    return ps.replace('ɾ', 'T')

def demote_later(ps):
    """Keep first primary, demote later primaries to secondary."""
    seen = False
    out = []
    for c in ps:
        if c == 'ˈ':
            out.append('ˌ' if seen else c)
            seen = True
        else:
            out.append(c)
    return ''.join(out)

def add_s(stem):
    last = stem[-1].rstrip('012')
    if last in ('S', 'Z', 'SH', 'ZH', 'CH', 'JH'):
        return stem + ('IH0', 'Z')
    return stem + ('Z',)

def add_ed(stem):
    last = stem[-1].rstrip('012')
    if last in ('T', 'D'):
        return stem + ('IH0', 'D')
    if last in ('P', 'K', 'F', 'TH', 'SH', 'S', 'CH'):
        return stem + ('T',)
    return stem + ('D',)

def add_ing(stem):
    return stem + ('IH0', 'NG')

def misaki_s(stem_ps):
    """misaki Lexicon._s (US English):
    if stem ends in voiceless consonant: + 's'
    if stem ends in sibilant (s, z, ʃ, ʒ, ʧ, ʤ): + 'ᵻz'
    else: + 'z'
    """
    if not stem_ps:
        return None
    last = stem_ps[-1]
    if last in 'ptkfθ':
        return stem_ps + 's'
    elif last in 'szʃʒʧʤ':
        return stem_ps + 'ᵻz'
    return stem_ps + 'z'


def split_compound(word):
    """Try splitting an OOV compound word into two known lexicon words.
    E.g. stonemasons -> stone + masons, treehouse -> tree + house."""
    w = word.lower()
    if len(w) < 6 or not w.isalpha():
        return None
    for i in range(3, len(w) - 2):
        w1 = w[:i]
        w2 = w[i:]
        ps1 = lexicon_lookup(w1) or cmu_lookup(w1)
        if ps1 is None:
            continue
        ps2 = lexicon_lookup(w2) or cmu_lookup(w2)
        if ps2 is None and w2.endswith('s') and not w2.endswith('ss'):
            ps2_stem = lexicon_lookup(w2[:-1]) or cmu_lookup(w2[:-1])
            if ps2_stem:
                ps2 = misaki_s(ps2_stem)
        if ps2:
            return demote_later(flap_all(ps1 + ps2))
    return None


L2S_CONSONANTS = {
    'th': 'θ', 'sh': 'ʃ', 'ch': 'ʧ', 'ph': 'f', 'qu': 'kw', 'wh': 'w',
    'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'ɡ', 'h': 'h',
    'j': 'ʤ', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p',
    'r': 'ɹ', 's': 's', 't': 't', 'v': 'v', 'w': 'w', 'x': 'ks',
    'z': 'z'
}
L2S_VOWELS = {
    'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɑ', 'u': 'ʌ', 'y': 'i'
}


def letter_to_sound(word):
    """Simple rule-based English letter-to-sound fallback for OOV words.
    Never drops a word from output."""
    w = word.lower()
    out = []
    i = 0
    first_vowel = True
    while i < len(w):
        if i + 2 <= len(w) and w[i:i+2] in L2S_CONSONANTS:
            out.append(L2S_CONSONANTS[w[i:i+2]])
            i += 2
        elif w[i] in L2S_CONSONANTS:
            out.append(L2S_CONSONANTS[w[i]])
            i += 1
        elif w[i] in L2S_VOWELS:
            if first_vowel:
                out.append('ˈ')
                first_vowel = False
            out.append(L2S_VOWELS[w[i]])
            i += 1
        elif w[i].isalpha():
            ps = spell_out(w[i])
            if ps:
                out.append(ps)
            i += 1
        else:
            i += 1
    return flap_all(''.join(out)) if out else None


def cmu_lookup(word):
    w = word.lower()
    phones = CMU.get(w)
    if phones is None:
        if len(w) > 2 and w.endswith('s') and not w.endswith('ss'):
            for stem in (w[:-1], w[:-2] if w.endswith('es') else None,
                         w[:-3] + 'y' if w.endswith('ies') else None):
                if stem and stem in CMU:
                    phones = add_s(CMU[stem])
                    break
        if phones is None and len(w) > 3 and w.endswith('d'):
            for stem in (w[:-1], w[:-2] if w.endswith('ed') and not w.endswith('eed') else None):
                if stem and stem in CMU:
                    phones = add_ed(CMU[stem])
                    break
        if phones is None and len(w) > 4 and w.endswith('ing'):
            for stem in (w[:-3], w[:-3] + 'e',
                         w[:-4] if re.search(r'([bcdgklmnprstvxz])\1ing$|cking$', w) else None):
                if stem and stem in CMU:
                    phones = add_ing(CMU[stem])
                    break
    if phones is None:
        return None
    ps = cmu_convert(phones)
    if ps is None:
        return None
    return demote_later(flap_all(flap(ps)))

def spell_out(word):
    """Letter-by-letter spelling using misaki's gold letter names."""
    out = []
    for c in word:
        if not c.isalpha():
            return None
        ps = LETTERS.get(c.upper())
        if ps is None:
            return None
        out.append(ps)
    return ' '.join(out) if out else None

LETTERS = {
    'A': 'ˈA', 'B': 'bˈi', 'C': 'sˈi', 'D': 'dˈi', 'E': 'ˈi', 'F': 'ˈɛf',
    'G': 'ʤˈi', 'H': 'ˈAʧ', 'I': 'ˈI', 'J': 'ʤˈA', 'K': 'kˈA', 'L': 'ˈɛl',
    'M': 'ˈɛm', 'N': 'ˈɛn', 'O': 'ˈO', 'P': 'pˈi', 'Q': 'kjˈu', 'R': 'ˈɑɹ',
    'S': 'ˈɛs', 'T': 'tˈi', 'U': 'jˈu', 'V': 'vˈi', 'W': 'dˈʌbᵊlju',
    'X': 'ˈɛks', 'Y': 'wˈI', 'Z': 'zˈi',
}

ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
        'eighty', 'ninety']

def num2words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (' ' + ONES[n % 10] if n % 10 else '')
    if n < 1000:
        return ONES[n // 100] + ' hundred' + (' ' + num2words(n % 100) if n % 100 else '')
    if n < 1000000:
        return num2words(n // 1000) + ' thousand' + (' ' + num2words(n % 1000) if n % 1000 else '')
    if n < 1000000000:
        return num2words(n // 1000000) + ' million' + (' ' + num2words(n % 1000000) if n % 1000000 else '')
    return str(n)

def year_words(n):
    # misaki (num2words to='year', with "and"/hyphens stripped):
    #   1000 -> one thousand, 1001 -> one thousand one, 1010 -> ten ten
    #   1100 -> eleven hundred, 1905 -> nineteen oh five
    #   2000 -> two thousand, 2001 -> two thousand one, 2010 -> twenty ten
    #   2100 -> twenty one hundred, 2101 -> twenty one oh one
    #   3000 -> three thousand, 9999 -> ninety nine ninety nine
    if 1000 <= n <= 9999:
        if n % 1000 == 0:
            return num2words(n)
        if n % 100 == 0:
            return num2words(n // 100) + ' hundred'
        if n % 1000 < 10:
            return num2words(n - n % 100) + ' ' + num2words(n % 100)
        h, r = divmod(n, 100)
        if r < 10:
            return num2words(h) + ' oh ' + num2words(r)
        return num2words(h) + ' ' + num2words(r)
    return num2words(n)

def currency_words(word):
    """misaki currency: $4.7 -> four dollars and seven cents;
    $0.99 -> ninety nine cents; $18 -> eighteen dollars."""
    word = word.replace(',', '')
    if '.' in word:
        whole, frac = word.split('.', 1)
    else:
        whole, frac = word, ''
    words = []
    whole_n = int(whole) if whole else 0
    frac_n = int(frac) if frac else 0
    if whole_n:
        words.extend(num2words(whole_n).split())
        words.append('dollars')
    if frac_n:
        if words:
            words.append('and')
        words.extend(num2words(frac_n).split())
        words.append('cents')
    if not words:
        words = ['zero', 'dollars']
    return words


def number_words(word):
    if re.fullmatch(r'\d+:\d+', word):
        # time: 9:30 -> nine thirty
        h, m = word.split(':')
        words = num2words(int(h)).split()
        if m == '00':
            words.append('hundred')
        else:
            words.extend(num2words(int(m)).split())
        return words
    if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', word):
        # date: 12/3/1905 -> twelve three nineteen oh five (no "slash")
        m_, d_, y_ = word.split('/')
        words = num2words(int(m_)).split()
        words.extend(num2words(int(d_)).split())
        words.extend(year_words(int(y_)).split())
        return words
    if re.fullmatch(r'\d{4}', word) and 1000 <= int(word) <= 9999:
        return year_words(int(word)).split()
    if re.fullmatch(r'\d+(,\d{3})*(\.\d+)?', word) and (',' in word or '.' in word):
        # thousands and/or decimal: "1,284.67" -> one thousand two hundred
        # eighty four point six seven; "42.50" -> forty two point five
        # (trailing zeros in the fraction are dropped, like num2words)
        if '.' in word:
            whole, frac = word.split('.', 1)
        else:
            whole, frac = word, None
        whole = whole.replace(',', '')
        words = num2words(int(whole)).split() if whole else []
        if frac is not None:
            frac = frac.rstrip('0')
            if frac:
                words.append('point')
                for c in frac:
                    words.append(ONES[int(c)])
        return words
    if re.fullmatch(r'\d+(,\d{3})*', word):
        return num2words(int(word.replace(',', ''))).split()
    if re.fullmatch(r'\d+', word):
        return num2words(int(word)).split()
    return None


ORDINAL_ONES = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth',
                'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh',
                'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
                'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth']
ORDINAL_TENS = ['', '', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth',
                'sixtieth', 'seventieth', 'eightieth', 'ninetieth']


def ordinal_num(n):
    """num2words(n, to='ordinal') with hyphens/and stripped, as misaki does."""
    if n < 20:
        return ORDINAL_ONES[n]
    if n < 100:
        if n % 10 == 0:
            return ORDINAL_TENS[n // 10]
        return TENS[n // 10] + ' ' + ORDINAL_ONES[n % 10]
    if n < 1000:
        if n % 100 == 0:
            return ONES[n // 100] + ' hundredth'
        return ONES[n // 100] + ' hundred ' + ordinal_num(n % 100)
    if n < 1000000:
        if n % 1000 == 0:
            return num2words(n // 1000) + ' thousandth'
        return num2words(n // 1000) + ' thousand ' + ordinal_num(n % 1000)
    return num2words(n) + 'th'


def ordinal_words(word):
    """26th -> twenty sixth, 8th -> eighth, 2nd -> second, 12th -> twelfth"""
    m = re.fullmatch(r'(\d+)(st|nd|rd|th)', word)
    if not m:
        return None
    return ordinal_num(int(m.group(1)))

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*(?:st|nd|rd|th)?|[;:,.!?—…\"“”()]|\S")
PUNCT = set(';:,.!?—…“”()')

def lexicon_lookup(word):
    v = GOLD.get(word) or GOLD.get(word.lower())
    if isinstance(v, dict):
        v = v.get('DEFAULT')
    if isinstance(v, str):
        return v
    v = SILVER.get(word) or SILVER.get(word.lower())
    if isinstance(v, dict):
        v = v.get('DEFAULT')
    if isinstance(v, str):
        return v
    return None

def cap_stress(ps):
    """misaki cap_stresses[0] = 0.5: add ˌ before first vowel if no marks."""
    if any(c in 'ˈˌ' for c in ps):
        return ps
    for i, c in enumerate(ps):
        if c in VOWELS:
            return ps[:i] + 'ˌ' + ps[i:]
    return ps

def next_word_phonemes(idx, tokens):
    """Phonemes of the next word-ish token (for the/a sound checks)."""
    for j in range(idx + 1, min(idx + 3, len(tokens))):
        t = tokens[j]
        if t in PUNCT or t in ('"', "''", '$', '-', '—'):
            continue
        if t == 'a' or t == 'A':
            return 'ɐ'
        if t == 'an':
            return 'ɐn'
        ps = lexicon_lookup(t)
        if ps is None:
            ps = cmu_lookup(t)
        if ps is None and t.isupper() and 1 < len(t) <= 6:
            ps = spell_out(t)
        if ps is None and re.search(r'\d', t):
            words = number_words(t)
            if words:
                ps = lexicon_lookup(words[0]) or cmu_lookup(words[0])
        return ps
    return None

def num_word_phonemes(w):
    """Phonemes for one word of a number expansion. misaki looks up "point"
    with stress -2 (all stress marks stripped)."""
    ps = lexicon_lookup(w)
    if ps is None:
        ps = cmu_lookup(w)
    if ps is None:
        return None
    if w == 'point':
        ps = ps.replace('ˈ', '').replace('ˌ', '')
    return ps


def g2p(text):
    text = text.lstrip('\ufeff')
    tokens = TOKEN_RE.findall(text)
    out = []          # list of (phonemes, space_before)
    oov = []
    sent_start = True
    prev_word = None
    for idx, tok in enumerate(tokens):
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else ''
        prev_tok = tokens[idx - 1] if idx > 0 else ''
        is_word_next = bool(re.match(r'[A-Za-z0-9]', nxt))
        if tok in PUNCT or tok in ('"', "''"):
            if out:
                out.append((tok, False))
            else:
                out.append((tok, True))
            continue
        if tok == '-' and idx + 1 < len(tokens) and re.fullmatch(r'\d+(,\d{3})*(\.\d+)?', tokens[idx + 1]):
            # negative number: -7 -> minus seven, -1,000 -> minus one thousand
            out.append(('mˈInəs', True))
            continue
        if tok == '-' or tok == '—':
            # hyphen between words: join without space
            continue  # handled below via compound check? simpler: skip
        # hyphenated compound: word-word
        if '-' in tok and re.fullmatch(r"[A-Za-z]+(-[A-Za-z]+)+", tok):
            parts = tok.split('-')
            pss = []
            ok = True
            for p in parts:
                ps = lexicon_lookup(p) or cmu_lookup(p)
                if ps is None:
                    ok = False
                    break
                pss.append(ps)
            if ok:
                joined = ''.join(pss)
                out.append((demote_later(joined), True))
                continue
            oov.append(tok)
            continue
        if tok == '$':
            # currency: attached to the next number token
            continue
        if tok == '/' and idx > 0 and idx + 1 < len(tokens) \
                and re.fullmatch(r'\d+', tokens[idx - 1]) \
                and re.fullmatch(r'\d+', tokens[idx + 1]):
            # date separator: 12/3/1905 -> just read the numbers
            continue
        if tok == '/' or tok == '+':
            # symbol words: slash, plus (misaki SYMBOLS/ADD_SYMBOLS)
            out.append(('slˈæʃ' if tok == '/' else 'plˈʌs', True))
            continue
        if tok == '@':
            out.append(('ˈæt', True))
            continue
        if tok == '%':
            # misaki SYMBOLS: % -> percent
            out.append(('pəɹsˈɛnt', True))
            continue
        if tok == "'":
            continue
        if re.search(r'\d', tok):
            # ordinal suffix: 26th -> twenty sixth
            m_ord = re.fullmatch(r'(\d+)(st|nd|rd|th)', tok)
            if m_ord:
                ow = ordinal_words(tok)
                if ow:
                    ok = True
                    for w in ow.split():
                        ps = lexicon_lookup(w)
                        if ps is None:
                            ps = cmu_lookup(w)
                        if ps is None:
                            ok = False
                            break
                        out.append((flap_all(ps), True))
                    if not ok:
                        oov.append(tok)
                    continue
            # currency prefix: $18 -> eighteen dollars, $4.7 -> four dollars
            # and seven cents, $0.99 -> ninety nine cents
            if prev_tok == '$':
                words = currency_words(tok)
                ok = True
                for w in words:
                    ps = num_word_phonemes(w)
                    if ps is None:
                        ok = False
                        break
                    out.append((flap_all(ps), True))
                if not ok:
                    oov.append(tok)
                continue
            words = number_words(tok)
            if words is None:
                oov.append(tok)
                continue
            ok = True
            for w in words:
                ps = num_word_phonemes(w)
                if ps is None:
                    ok = False
                    break
                out.append((flap_all(ps), True))
            if not ok:
                oov.append(tok)
            continue
        # function-word context rules
        low = tok.lower()
        if tok == 'a' or (tok == 'A' and not sent_start):
            out.append(('ɐ', True))
            continue
        if tok == 'A' and sent_start:
            # 'A' at sentence start: article if a word follows, letter if not
            if is_word_next and nxt and nxt[0].islower():
                out.append(('ɐ', True))
            else:
                out.append(('ˈA', True))
            continue
        if low == 'an':
            out.append(('ɐn', True))
            continue
        if low == 'the':
            # misaki: ði before vowel-SOUND-initial word, ðə otherwise
            nxt_ps = next_word_phonemes(idx, tokens)
            nxt_clean = nxt_ps.lstrip('ˈˌ') if nxt_ps else ''
            if nxt_clean and nxt_clean[0] in 'AEIOUaeiɔɑʊɪɛəɐᵻæ':
                out.append(('ði', True))
            else:
                out.append(('ðə', True))
            continue
        if low == 'this':
            out.append(('ðɪs', True))
            continue
        if low == 'to':
            # misaki: {None: 'tu', False: 'tə', True: 'tʊ'}[ctx.future_vowel]
            nxt_ps = next_word_phonemes(idx, tokens)
            if nxt_ps:
                nxt_clean = nxt_ps.lstrip('ˈˌ')
                if nxt_clean and nxt_clean[0] in 'AEIOUaeiɔɑʊɪɛəɐᵻæ':
                    out.append(('tʊ', True))
                else:
                    out.append(('tə', True))
            else:
                out.append(('tu', True))
            continue
        if low == 'used':
            nxt_tok = tokens[idx + 1].lower() if idx + 1 < len(tokens) else ''
            if nxt_tok == 'to':
                out.append(('jˈust', True))
                continue
        if tok.startswith('-') and re.fullmatch(r'\d+', tok[1:]):
            # negative number: -7 -> minus seven
            out.append(('mˈInəs', True))
            words = num2words(int(tok[1:])).split()
            ok = True
            for w in words:
                ps = lexicon_lookup(w)
                if ps is None:
                    ps = cmu_lookup(w)
                if ps is None:
                    ok = False
                    break
                out.append((ps, True))
            if not ok:
                oov.append(tok)
            continue
        if low == 'i':
            out.append(('ˌI', True))
            continue
        if low == 'in':
            # misaki: ˈɪn when stressed (sentence-final / before punct), ɪn otherwise
            nxt_ps = next_word_phonemes(idx, tokens)
            if nxt_ps is None:
                out.append(('ˈɪn', True))
            else:
                out.append(('ɪn', True))
            continue
        if low == 'there':
            # misaki: ðˌɛɹ DEFAULT, ðˈɛɹ with None tag (sentence-final or copula)
            nxt_ps = next_word_phonemes(idx, tokens)
            if nxt_ps is None:
                out.append(('ðˈɛɹ', True))
            else:
                out.append(('ðˌɛɹ', True))
            continue
        if low == 'am':
            out.append(('ɐm', True))
            continue
        if tok == 'That' and sent_start:
            out.append(('ðˈæt', True))
            sent_start = False
            continue
        ps = lexicon_lookup(tok)
        # possessive 's against misaki lexicon first: boss's -> boss + ᵻz
        if ps is None and tok.endswith("'s") and len(tok) > 2:
            stem = tok[:-2]
            stem_ps = lexicon_lookup(stem)
            if stem_ps:
                ps = misaki_s(stem_ps)
        # plural possessive: teachers' -> teachers
        if ps is None and tok.endswith("s'") and len(tok) > 2:
            stem = tok[:-1]
            stem_ps = lexicon_lookup(stem)
            if stem_ps:
                ps = stem_ps
        # cmu fallback
        if ps is None:
            ps = cmu_lookup(tok)
        # cmu-based possessive
        if ps is None and tok.endswith("'s") and len(tok) > 2:
            stem = tok[:-2]
            stem_ps = cmu_lookup(stem)
            if stem_ps:
                ps = misaki_s(stem_ps)
        if ps is None and tok.endswith("s'") and len(tok) > 2:
            stem = tok[:-1]
            stem_ps = cmu_lookup(stem)
            if stem_ps:
                ps = stem_ps
        # compound splitting: stonemasons -> stone + masons
        if ps is None and len(tok) >= 6 and tok.isalpha():
            ps = split_compound(tok)
        # acronym spelling: letters joined without spaces, stress kept
        if tok.isupper() and 1 < len(tok) <= 6:
            ps = spell_out(tok)
            if ps is not None:
                ps = ps.replace(' ', '')
        # OOV fallback: rule-based letter-to-sound (never drop a word!)
        if ps is None:
            oov.append(tok)
            ps = letter_to_sound(tok)
        if ps is None:
            continue
        # paradee final pass: all ɾ -> T (applies to lexicon entries too)
        ps = flap_all(ps)
        # cap stress for capitalized non-special words (not sentence-initial
        # special cases; misaki applies to any capitalized word)
        if tok[0].isupper() and tok[1:].islower() and not any(c in 'ˈˌ' for c in ps):
            ps = cap_stress(ps)
        out.append((ps, True))
        sent_start = False
    # assemble with spacing: no space before punct
    parts = []
    for ps, sp in out:
        if parts and sp and not (len(ps) == 1 and ps in PUNCT):
            parts.append(' ')
        parts.append(ps)
    return (''.join(parts), oov)

def lev(a, b):
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]

HETERONYMS = {
    'lead', 'read', 'live', 'wind', 'bow', 'close', 'does', 'minute',
    'object', 'present', 'produce', 'project', 'record', 'subject', 'use',
    'bass', 'dove', 'tear', 'wound', 'content', 'refuse',
}

def main():
    refs = json.load(open(os.path.join(HERE, 'ref.json'), encoding='utf-8'))
    n = len(refs)
    exact = 0
    total_ref = 0
    total_err = 0
    causes = Counter()
    examples = []
    t0 = time.perf_counter()
    for r in refs:
        text, ref = r['text'], r['phonemes']
        got, oov = g2p(text)
        if oov:
            causes['oov'] += 1
            total_ref += len(ref)
            total_err += len(ref)
            if len(examples) < 60:
                examples.append(('OOV ' + ','.join(oov), text[:50], ref, got if got else ''))
            continue
        if got == ref:
            exact += 1
            continue
        dist = lev(got, ref)
        total_ref += len(ref)
        total_err += dist
        toks = set(t.lower() for t in TOKEN_RE.findall(text))
        if any(re.fullmatch(r'\d+([.,]\d+)*', t) for t in toks):
            causes['numbers'] += 1
        elif any(t in HETERONYMS for t in toks):
            causes['heteronym_default'] += 1
        else:
            causes['mapping_or_stress'] += 1
        if len(examples) < 60 and causes['mapping_or_stress'] < 12:
            examples.append(('DIFF', text[:50], ref, got))
    dt = time.perf_counter() - t0
    print(f'sentences: {n}')
    print(f'exact match: {exact} ({100 * exact / n:.1f}%)')
    print(f'PER: {100 * total_err / max(1, total_ref):.2f}%')
    print(f'breakdown: {dict(causes)}')
    print(f'g2p time: {dt * 1000 / n:.2f} ms/sentence')
    print()
    for kind, t, ref, got in examples[:20]:
        print(f'{kind}: {t}')
        print(f'  ref: {ref[:90]}')
        print(f'  got: {got[:90]}')
        print()

if __name__ == '__main__':
    main()
