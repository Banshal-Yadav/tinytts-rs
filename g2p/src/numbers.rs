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

/// proto3 year_words: 1000..=2099 read as a year.
pub fn year_words(n: u64) -> String {
    if (1000..=2099).contains(&n) {
        let h = n / 100;
        let r = n % 100;
        if r == 0 {
            return format!("{} hundred", num2words(h));
        }
        if n < 2000 {
            if r < 10 {
                return format!("{} oh {}", num2words(h), num2words(r));
            }
            return format!("{} {}", num2words(h), num2words(r));
        }
        if r < 10 {
            return format!("{} oh {}", num2words(h), num2words(r));
        }
        // 2050 -> "twenty fifty", 2019 -> "twenty nineteen"
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
    // year: 4 digits in 1000..=2099
    if word.len() == 4 && is_digits(word) {
        let n: u64 = word.parse().ok()?;
        if (1000..=2099).contains(&n) {
            return Some(year_words(n).split(' ').map(String::from).collect());
        }
    }
    // thousands and/or decimal: "1,284.67" -> one thousand two hundred
    // eighty four point six seven; "42.50" -> forty two point five zero
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
            words.push("point".to_string());
            for c in f.chars() {
                words.push(ONES[c.to_digit(10)? as usize].to_string());
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

/// proto3 ordinal_words: 26th -> twenty sixth, 8th -> eighth, 2nd -> second.
pub fn ordinal_words(word: &str) -> Option<String> {
    let (num, suffix) = split_ordinal(word)?;
    let n: u64 = num.parse().ok()?;
    let irreg = |k: u64| -> Option<&'static str> {
        match k {
            1 => Some("first"),
            2 => Some("second"),
            3 => Some("third"),
            5 => Some("fifth"),
            8 => Some("eighth"),
            9 => Some("ninth"),
            12 => Some("twelfth"),
            _ => None,
        }
    };
    if matches!(n % 100, 11..=13) {
        return Some(format!("{}th", num2words(n)));
    }
    if n < 10 {
        if let Some(w) = irreg(n % 10) {
            return Some(w.to_string());
        }
    }
    if n > 20 {
        if let Some(w) = irreg(n % 10) {
            return Some(format!("{} {}", num2words(n - n % 10), w));
        }
    }
    let suf = match n % 10 {
        1 => "st",
        2 => "nd",
        3 => "rd",
        _ => "th",
    };
    let _ = suffix;
    // 40 -> forty + th = fortieth (drop the y)
    let mut base = num2words(n);
    if base.ends_with('y') {
        base.pop();
        base.push_str("ie");
    }
    Some(format!("{}{}", base, suf))
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
        assert_eq!(year_words(2019), "twenty nineteen");
        assert_eq!(year_words(1905), "nineteen oh five");
        assert_eq!(year_words(2000), "twenty hundred");
        assert_eq!(year_words(2050), "twenty fifty");
        assert_eq!(year_words(1945), "nineteen forty five");
    }

    #[test]
    fn ordinals() {
        assert_eq!(ordinal_words("26th"), Some("twenty sixth".to_string()));
        assert_eq!(ordinal_words("8th"), Some("eighth".to_string()));
        assert_eq!(ordinal_words("2nd"), Some("second".to_string()));
        assert_eq!(ordinal_words("40th"), Some("fortieth".to_string()));
        assert_eq!(ordinal_words("13th"), Some("thirteenth".to_string()));
    }

    #[test]
    fn decimals_and_thousands() {
        assert_eq!(
            number_words("42.50"),
            Some("forty two point five zero".split(' ').map(String::from).collect())
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
