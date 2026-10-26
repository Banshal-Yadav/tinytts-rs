//! Number, date, currency, ordinal, decimal, and time expansion, ported from
//! the validated Python prototype (g2p-spike/proto3.py: num2words, year_words,
//! number_words, ordinal_words). The differential test (scripts/diff_g2p.py)
//! enforces byte-exact agreement with that reference.

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// proto3 num2words: cardinal words for a non-negative integer.
pub fn num2words(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let t = TENS[(n / 10) as usize];
        return if n.is_multiple_of(10) {
            t.to_string()
        } else {
            format!("{} {}", t, ONES[(n % 10) as usize])
        };
    }
    if n < 1000 {
        let mut s = format!("{} hundred", ONES[(n / 100) as usize]);
        if !n.is_multiple_of(100) {
            s.push(' ');
            s.push_str(&num2words(n % 100));
        }
        return s;
    }
    if n < 1_000_000 {
        let mut s = format!("{} thousand", num2words(n / 1000));
        if !n.is_multiple_of(1000) {
            s.push(' ');
            s.push_str(&num2words(n % 1000));
        }
        return s;
    }
    if n < 1_000_000_000 {
        let mut s = format!("{} million", num2words(n / 1_000_000));
        if !n.is_multiple_of(1_000_000) {
            s.push(' ');
            s.push_str(&num2words(n % 1_000_000));
        }
        return s;
    }
    n.to_string()
}

/// Year reading, matching misaki (num2words to='year', "and"/hyphens
/// stripped): 1000 -> one thousand, 1001 -> one thousand one, 1010 -> ten ten,
/// 1100 -> eleven hundred, 1905 -> nineteen oh five, 2100 -> twenty one
/// hundred, 2101 -> twenty one oh one, 3000 -> three thousand.
pub fn year_words(n: u64) -> String {
    if (1000..=9999).contains(&n) {
        if n.is_multiple_of(1000) {
            return num2words(n);
        }
        if n.is_multiple_of(100) {
            return format!("{} hundred", num2words(n / 100));
        }
        if n % 1000 < 10 {
            return format!("{} {}", num2words(n - n % 100), num2words(n % 100));
        }
        let h = n / 100;
        let r = n % 100;
        if r < 10 {
            return format!("{} oh {}", num2words(h), num2words(r));
        }
        return format!("{} {}", num2words(h), num2words(r));
    }
    num2words(n)
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// proto3 number_words: returns the word list for a numeric token, or None.
pub fn number_words(word: &str) -> Option<Vec<String>> {
    // time: 9:30 -> nine thirty
    if let Some((h, m)) = word.split_once(':') {
        if is_digits(h) && is_digits(m) {
            let mut words: Vec<String> = num2words(h.parse().ok()?).split(' ').map(String::from).collect();
            if m == "00" {
                words.push("hundred".to_string());
            } else {
                words.extend(num2words(m.parse().ok()?).split(' ').map(String::from));
            }
            return Some(words);
        }
    }
    // date: 12/3/1905 -> twelve three nineteen oh five
    let slash: Vec<&str> = word.split('/').collect();
    if slash.len() == 3 && is_digits(slash[0]) && is_digits(slash[1]) && slash[2].len() == 4 && is_digits(slash[2]) {
        let mut words: Vec<String> = num2words(slash[0].parse().ok()?).split(' ').map(String::from).collect();
        words.extend(num2words(slash[1].parse().ok()?).split(' ').map(String::from));
        words.extend(year_words(slash[2].parse().ok()?).split(' ').map(String::from));
        return Some(words);
    }
    // year: 4 digits in 1000..=9999
    if word.len() == 4 && is_digits(word) {
        let n: u64 = word.parse().ok()?;
        if (1000..=9999).contains(&n) {
            return Some(year_words(n).split(' ').map(String::from).collect());
        }
    }
    // thousands and/or decimal: "1,284.67" -> one thousand two hundred
    // eighty four point six seven; "42.50" -> forty two point five
    // (trailing zeros in the fraction are dropped, like num2words)
    if (word.contains(',') || word.contains('.')) && is_thousands_decimal(word) {
        let (whole, frac) = match word.split_once('.') {
            Some((w, f)) => (w, Some(f)),
            None => (word, None),
        };
        let whole_clean = whole.replace(',', "");
        let mut words: Vec<String> = if whole_clean.is_empty() {
            Vec::new()
        } else {
            num2words(whole_clean.parse().ok()?).split(' ').map(String::from).collect()
        };
        if let Some(f) = frac {
            let f = f.trim_end_matches('0');
            if !f.is_empty() {
                words.push("point".to_string());
                for c in f.chars() {
                    words.push(ONES[c.to_digit(10)? as usize].to_string());
                }
            }
        }
        return Some(words);
    }
    // thousands: 1,284 -> one thousand two hundred eighty four
    if is_thousands(word) {
        let n: u64 = word.replace(',', "").parse().ok()?;
        return Some(num2words(n).split(' ').map(String::from).collect());
    }
    // plain integer
    if is_digits(word) {
        return Some(num2words(word.parse().ok()?).split(' ').map(String::from).collect());
    }
    None
}

/// proto3 regex `\d+(,\d{3})*(\.\d+)?`: digits with optional 3-digit comma
/// groups and an optional decimal part.
fn is_thousands_decimal(word: &str) -> bool {
    let (whole, frac) = match word.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (word, None),
    };
    if let Some(f) = frac {
        if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
    }
    is_thousands(whole)
}

/// proto3 regex `\d+(,\d{3})*`: digits with optional 3-digit comma groups.
fn is_thousands(word: &str) -> bool {
    let parts: Vec<&str> = word.split(',').collect();
    if parts.is_empty() || !is_digits(parts[0]) {
        return false;
    }
    parts[1..].iter().all(|p| p.len() == 3 && is_digits(p))
}

const ORDINAL_ONES: [&str; 20] = [
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth",
];
const ORDINAL_TENS: [&str; 10] = [
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
    "eightieth", "ninetieth",
];

/// num2words(n, to='ordinal') with hyphens/and stripped, matching misaki.
pub fn ordinal_num(n: u64) -> String {
    if n < 20 {
        return ORDINAL_ONES[n as usize].to_string();
    }
    if n < 100 {
        if n.is_multiple_of(10) {
            return ORDINAL_TENS[(n / 10) as usize].to_string();
        }
        return format!("{} {}", TENS[(n / 10) as usize], ORDINAL_ONES[(n % 10) as usize]);
    }
    if n < 1000 {
        if n.is_multiple_of(100) {
            return format!("{} hundredth", ONES[(n / 100) as usize]);
        }
        return format!("{} hundred {}", ONES[(n / 100) as usize], ordinal_num(n % 100));
    }
    if n < 1_000_000 {
        if n.is_multiple_of(1000) {
            return format!("{} thousandth", num2words(n / 1000));
        }
        return format!("{} thousand {}", num2words(n / 1000), ordinal_num(n % 1000));
    }
    format!("{}th", num2words(n))
}

/// proto3 ordinal_words: 26th -> twenty sixth, 8th -> eighth, 2nd -> second, 12th -> twelfth.
pub fn ordinal_words(word: &str) -> Option<String> {
    let (num, _) = split_ordinal(word)?;
    let n: u64 = num.parse().ok()?;
    Some(ordinal_num(n))
}

/// misaki currency: $4.7 -> four dollars and seven cents;
/// $0.99 -> ninety nine cents; $18 -> eighteen dollars.
pub fn currency_words(word: &str) -> Vec<String> {
    let clean = word.replace(',', "");
    let (whole, frac) = match clean.split_once('.') {
        Some((w, f)) => (w, f),
        None => (clean.as_str(), ""),
    };
    let whole_n: u64 = whole.parse().unwrap_or(0);
    let frac_n: u64 = frac.parse().unwrap_or(0);
    let mut words = Vec::new();
    if whole_n > 0 {
        words.extend(num2words(whole_n).split(' ').map(String::from));
        words.push("dollars".to_string());
    }
    if frac_n > 0 {
        if !words.is_empty() {
            words.push("and".to_string());
        }
        words.extend(num2words(frac_n).split(' ').map(String::from));
        words.push("cents".to_string());
    }
    if words.is_empty() {
        words.push("zero".to_string());
        words.push("dollars".to_string());
    }
    words
}

/// Split "26th" into ("26", "th"); None if not `<digits><st|nd|rd|th>`.
fn split_ordinal(word: &str) -> Option<(&str, &str)> {
    for suf in ["st", "nd", "rd", "th"] {
        if let Some(num) = word.strip_suffix(suf) {
            if is_digits(num) {
                return Some((num, suf));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals() {
        assert_eq!(num2words(0), "zero");
        assert_eq!(num2words(19), "nineteen");
        assert_eq!(num2words(20), "twenty");
        assert_eq!(num2words(42), "forty two");
        assert_eq!(num2words(100), "one hundred");
        assert_eq!(num2words(1284), "one thousand two hundred eighty four");
        assert_eq!(num2words(45000), "forty five thousand");
    }

    #[test]
    fn years() {
        assert_eq!(year_words(1000), "one thousand");
        assert_eq!(year_words(1001), "one thousand one");
        assert_eq!(year_words(1905), "nineteen oh five");
        assert_eq!(year_words(2000), "two thousand");
        assert_eq!(year_words(2005), "two thousand five");
        assert_eq!(year_words(2010), "twenty ten");
        assert_eq!(year_words(2019), "twenty nineteen");
        assert_eq!(year_words(2050), "twenty fifty");
        assert_eq!(year_words(2100), "twenty one hundred");
        assert_eq!(year_words(1945), "nineteen forty five");
    }

    #[test]
    fn ordinals() {
        assert_eq!(ordinal_words("26th"), Some("twenty sixth".to_string()));
        assert_eq!(ordinal_words("8th"), Some("eighth".to_string()));
        assert_eq!(ordinal_words("2nd"), Some("second".to_string()));
        assert_eq!(ordinal_words("12th"), Some("twelfth".to_string()));
        assert_eq!(ordinal_words("40th"), Some("fortieth".to_string()));
        assert_eq!(ordinal_words("13th"), Some("thirteenth".to_string()));
        assert_eq!(ordinal_words("100th"), Some("one hundredth".to_string()));
    }

    #[test]
    fn decimals_and_thousands() {
        assert_eq!(
            number_words("42.50"),
            Some("forty two point five".split(' ').map(String::from).collect())
        );
        assert_eq!(
            number_words("4.0"),
            Some("four".split(' ').map(String::from).collect())
        );
        assert_eq!(
            number_words("100.00"),
            Some("one hundred".split(' ').map(String::from).collect())
        );
        assert_eq!(
            number_words("1,284.67"),
            Some("one thousand two hundred eighty four point six seven".split(' ').map(String::from).collect())
        );
        assert_eq!(
            number_words("1,284"),
            Some("one thousand two hundred eighty four".split(' ').map(String::from).collect())
        );
    }

    #[test]
    fn currency() {
        assert_eq!(
            currency_words("18"),
            vec!["eighteen", "dollars"]
        );
        assert_eq!(
            currency_words("4.7"),
            vec!["four", "dollars", "and", "seven", "cents"]
        );
        assert_eq!(
            currency_words("0.99"),
            vec!["ninety", "nine", "cents"]
        );
    }

    #[test]
    fn times_and_dates() {
        assert_eq!(
            number_words("9:30"),
            Some("nine thirty".split(' ').map(String::from).collect())
        );
        assert_eq!(
            number_words("12/3/1905"),
            Some("twelve three nineteen oh five".split(' ').map(String::from).collect())
        );
    }
}
