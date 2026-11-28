//! The clock and the spellings for the times an index's files record, and
//! the date a daily log file is named by.
//!
//! [`Timestamp`] is a UTC instant, held as whole seconds from
//! 1970-01-01T00:00:00Z, negative before it, and the nanoseconds past that
//! second. [`Timestamp::now`] reads the system clock, and `From<SystemTime>`
//! converts any system time, one before 1970 included, which is the seam a
//! test gives the clock a time through. Dates are those of the proleptic
//! Gregorian calendar. Nothing here reads a time zone, and nothing but `std`
//! is used.
//!
//! # The spellings
//!
//! Each of the three below keeps its spelling byte for byte, since the
//! saved format records the first as text and the other two name folders
//! and files.
//!
//! - [`Timestamp::to_rfc3339`], such as `2026-11-27T06:27:49.767390800+00:00`,
//!   for each time that `manifest.json`, `quantization.json`, a training
//!   record of the journal, a lock file and `repair.json` record.
//! - [`Timestamp::to_compact`], such as `20261127T062749767Z`, which names a
//!   quarantine folder.
//! - [`Timestamp::to_date`], such as `2026-11-27`, which names a daily log
//!   file as tracing-appender names it.
//!
//! The year is four digits from 0 to 9999, and a sign before four or more
//! outside them. In the years 0 to 9999 two spellings of one kind sort as
//! text in the order of their instants, which a comparison of two `saved_at`
//! values and the order of the quarantine folders rest on.
//!
//! # What a load reads
//!
//! [`Timestamp::parse_rfc3339`] reads a date-time by the grammar of RFC 3339
//! section 5.6 and its two notes, which allow a lower case `t` and `z` and a
//! space in place of the `T`. Fraction digits past the ninth are read and
//! dropped. It also reads two forms RFC 3339 does not give, being U+2212 as
//! the minus of an offset, and a second of 60 in any minute, which names the
//! instant of the first second of the next minute. A load accepts exactly
//! this set and refuses every other text, so a directory a load opens keeps
//! opening.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Nanoseconds in a second.
const NANOS_PER_SEC: u32 = 1_000_000_000;

/// Seconds in a day. The clock counts no leap second, as `SystemTime`
/// counts none.
const SECS_PER_DAY: i64 = 86_400;

/// Days in 400 years, the period the Gregorian leap rule repeats on.
const DAYS_PER_400_YEARS: i64 = 146_097;

/// Days in 100 years that hold no 29 February of a year divisible by 400.
const DAYS_PER_100_YEARS: i64 = 36_524;

/// Days in 4 years that hold one 29 February.
const DAYS_PER_4_YEARS: i64 = 1_461;

/// Days in a year that holds no 29 February.
const DAYS_PER_YEAR: i64 = 365;

/// Days from 0000-03-01 to 1970-01-01.
///
/// The arithmetic counts each year from 1 March, so a 29 February is the
/// last day of the year it ends, and a period of 400 such years starts at
/// 0000-03-01.
const DAYS_FROM_YEAR_ZERO_TO_EPOCH: i64 = 719_468;

/// The day each month starts on in a year counted from 1 March, March first
/// and February last.
const MONTH_STARTS: [i64; 12] = [0, 31, 61, 92, 122, 153, 184, 214, 245, 275, 306, 337];

/// The calendar month of each entry of [`MONTH_STARTS`].
const MONTHS: [u8; 12] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2];

/// A UTC instant, as whole seconds from 1970-01-01T00:00:00Z, negative
/// before it, and the nanoseconds past that second.
///
/// Two timestamps order as their instants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    secs: i64,
    /// Below one second.
    nanos: u32,
}

/// The calendar fields of a UTC instant.
struct Fields {
    year: i64,
    month: u8,
    day: u8,
    hour: i64,
    minute: i64,
    second: i64,
}

impl Timestamp {
    /// The system clock's time, read once.
    pub fn now() -> Self {
        Self::from(SystemTime::now())
    }

    /// The instant in RFC 3339 at UTC, such as
    /// `2026-11-27T06:27:49.767390800+00:00`.
    ///
    /// The fraction is left out on a whole second, and otherwise has the
    /// fewest of three, six or nine digits that hold it. The offset is
    /// `+00:00`.
    pub fn to_rfc3339(self) -> String {
        let at = self.fields();
        format!(
            "{}-{:02}-{:02}T{:02}:{:02}:{:02}{}+00:00",
            Year(at.year),
            at.month,
            at.day,
            at.hour,
            at.minute,
            at.second,
            Fraction(self.nanos)
        )
    }

    /// The instant at UTC to the millisecond, such as `20261127T062749767Z`,
    /// with the nanoseconds past the millisecond cut.
    ///
    /// It is 19 bytes in the years 0 to 9999.
    pub fn to_compact(self) -> String {
        let at = self.fields();
        format!(
            "{}{:02}{:02}T{:02}{:02}{:02}{:03}Z",
            Year(at.year),
            at.month,
            at.day,
            at.hour,
            at.minute,
            at.second,
            self.nanos / 1_000_000
        )
    }

    /// The instant's UTC date, such as `2026-11-27`.
    pub fn to_date(self) -> String {
        let at = self.fields();
        format!("{}-{:02}-{:02}", Year(at.year), at.month, at.day)
    }

    /// The instant an RFC 3339 date-time names, or `None` where `text` is not
    /// one a load reads.
    ///
    /// `text` is a date, then `T`, `t` or a space, a time of day, an optional
    /// fraction of one or more digits, and an offset, being `Z`, `z`, or `+`,
    /// `-` or U+2212 followed by hours and minutes up to 23:59, and nothing
    /// else. The date is a four digit year and a two digit month and day
    /// that the month holds, and the time is two digit hours to 23, minutes
    /// to 59 and seconds to 60. See the module doc.
    pub fn parse_rfc3339(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        // The date, the separator and the time of day are the first 19
        // bytes, each of them ASCII.
        let head = bytes.get(..19)?;
        let year = i64::from(two_digits(&head[0..2])?) * 100 + i64::from(two_digits(&head[2..4])?);
        let month = two_digits(&head[5..7])?;
        let day = two_digits(&head[8..10])?;
        let hour = two_digits(&head[11..13])?;
        let minute = two_digits(&head[14..16])?;
        let second = two_digits(&head[17..19])?;
        let separated = head[4] == b'-'
            && head[7] == b'-'
            && matches!(head[10], b'T' | b't' | b' ')
            && head[13] == b':'
            && head[16] == b':';
        if !separated
            || !(1..=days_in_month(year, month)).contains(&day)
            || hour > 23
            || minute > 59
            || second > 60
        {
            return None;
        }
        // A fraction's first nine digits are its nanoseconds, and the
        // digits after them are dropped.
        let mut end = 19;
        let mut nanos = 0;
        if bytes.get(end) == Some(&b'.') {
            let digits = bytes[end + 1..]
                .iter()
                .take_while(|byte| byte.is_ascii_digit())
                .count();
            if digits == 0 {
                return None;
            }
            let fraction = &bytes[end + 1..end + 1 + digits];
            nanos = (0..9).fold(0, |value, place| {
                value * 10
                    + fraction
                        .get(place)
                        .map_or(0, |digit| u32::from(digit - b'0'))
            });
            end += 1 + digits;
        }
        let east = offset_east(text.get(end..)?)?;
        // A second of 60 counts as one more second than 59, which is the
        // first second of the next minute.
        let local = days_from_civil(year, month, day) * SECS_PER_DAY
            + i64::from(hour) * 3_600
            + i64::from(minute) * 60
            + i64::from(second);
        Some(Timestamp {
            secs: local - east,
            nanos,
        })
    }

    /// The calendar fields of the instant.
    fn fields(self) -> Fields {
        let (year, month, day) = civil_from_days(self.secs.div_euclid(SECS_PER_DAY));
        let of_day = self.secs.rem_euclid(SECS_PER_DAY);
        Fields {
            year,
            month,
            day,
            hour: of_day / 3_600,
            minute: of_day / 60 % 60,
            second: of_day % 60,
        }
    }
}

impl From<SystemTime> for Timestamp {
    /// The instant `time` names, before 1970 as after it. A time further
    /// from 1970 than `i64` seconds reach is held at the end of that range.
    fn from(time: SystemTime) -> Self {
        match time.duration_since(UNIX_EPOCH) {
            Ok(after) => Timestamp {
                secs: i64::try_from(after.as_secs()).unwrap_or(i64::MAX),
                nanos: after.subsec_nanos(),
            },
            Err(before) => {
                let before = before.duration();
                let secs = i64::try_from(before.as_secs()).map_or(i64::MIN, |secs| -secs);
                match before.subsec_nanos() {
                    0 => Timestamp { secs, nanos: 0 },
                    // A fraction before a whole second is taken from the
                    // second before it: one second less, and the nanoseconds
                    // the fraction leaves of that second.
                    nanos => Timestamp {
                        secs: secs.saturating_sub(1),
                        nanos: NANOS_PER_SEC - nanos,
                    },
                }
            }
        }
    }
}

/// A year as the spellings write it: four digits from 0 to 9999, and a sign
/// before four or more outside them.
struct Year(i64);

impl fmt::Display for Year {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if (0..=9_999).contains(&self.0) {
            write!(f, "{:04}", self.0)
        } else {
            write!(f, "{:+05}", self.0)
        }
    }
}

/// A fraction of a second as an RFC 3339 spelling writes it: nothing for
/// none, and otherwise a point and the fewest of three, six or nine digits
/// that hold it.
struct Fraction(u32);

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nanos = self.0;
        if nanos == 0 {
            Ok(())
        } else if nanos.is_multiple_of(1_000_000) {
            write!(f, ".{:03}", nanos / 1_000_000)
        } else if nanos.is_multiple_of(1_000) {
            write!(f, ".{:06}", nanos / 1_000)
        } else {
            write!(f, ".{nanos:09}")
        }
    }
}

/// The value of two ASCII digits, or `None` where either is not one.
fn two_digits(pair: &[u8]) -> Option<u8> {
    match *pair {
        [tens @ b'0'..=b'9', ones @ b'0'..=b'9'] => Some((tens - b'0') * 10 + (ones - b'0')),
        _ => None,
    }
}

/// The seconds east of UTC that `rest` names, or `None` where it is not an
/// offset with nothing after it.
fn offset_east(rest: &str) -> Option<i64> {
    if rest == "Z" || rest == "z" {
        return Some(0);
    }
    let (east, hours_and_minutes) = match rest.strip_prefix('+') {
        Some(after) => (true, after),
        None => (false, rest.strip_prefix(['-', '\u{2212}'])?),
    };
    let &[hour_tens, hour_ones, b':', minute_tens, minute_ones] = hours_and_minutes.as_bytes()
    else {
        return None;
    };
    let hours = two_digits(&[hour_tens, hour_ones])?;
    let minutes = two_digits(&[minute_tens, minute_ones])?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    let secs = i64::from(hours) * 3_600 + i64::from(minutes) * 60;
    Some(if east { secs } else { -secs })
}

/// Whether `year` has a 29 February in the proleptic Gregorian calendar.
fn is_leap_year(year: i64) -> bool {
    year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

/// The days of `month` in `year`, or 0 where `month` is not 1 to 12.
fn days_in_month(year: i64, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// The days from 1970-01-01 to the date `year-month-day`, negative before
/// it. `month` is 1 to 12, and `day` is 1 to the days of that month.
fn days_from_civil(year: i64, month: u8, day: u8) -> i64 {
    // The year counted from 1 March that holds the date, and the month's
    // place in that year.
    let (year, place) = if month >= 3 {
        (year, month - 3)
    } else {
        (year - 1, month + 9)
    };
    let years = year.rem_euclid(400);
    // A year counted from 1 March ends on a 29 February where the calendar
    // year after it is a leap year, so the years of a period before this one
    // end on years / 4 of them, less years / 100.
    let day_of_period = years * DAYS_PER_YEAR + years / 4 - years / 100
        + MONTH_STARTS[usize::from(place)]
        + i64::from(day)
        - 1;
    year.div_euclid(400) * DAYS_PER_400_YEARS + day_of_period - DAYS_FROM_YEAR_ZERO_TO_EPOCH
}

/// The date `days` after 1970-01-01, which [`days_from_civil`] undoes: the
/// year, the month from 1 to 12 and the day from 1 to 31.
fn civil_from_days(days: i64) -> (i64, u8, u8) {
    let since = days + DAYS_FROM_YEAR_ZERO_TO_EPOCH;
    let periods = since.div_euclid(DAYS_PER_400_YEARS);
    let mut rest = since.rem_euclid(DAYS_PER_400_YEARS);
    // A period's last century is a day longer than the others, since it ends
    // on the 29 February of a year divisible by 400, so its last day stays
    // in it.
    let centuries = (rest / DAYS_PER_100_YEARS).min(3);
    rest -= centuries * DAYS_PER_100_YEARS;
    // A century's last four years are a day shorter than the others, except
    // in a period's last century, so no day of a century passes its 25th
    // group.
    let groups = rest / DAYS_PER_4_YEARS;
    rest -= groups * DAYS_PER_4_YEARS;
    // A group's last year is a day longer than the others, since it ends on
    // a 29 February, so its last day stays in it.
    let years = (rest / DAYS_PER_YEAR).min(3);
    rest -= years * DAYS_PER_YEAR;
    // `rest` is now the day of a year counted from 1 March, 0 to 365, and
    // the first month start is 0, so one start at least is at or before it.
    let place = MONTH_STARTS.partition_point(|&start| start <= rest) - 1;
    let month = MONTHS[place];
    #[expect(clippy::cast_possible_truncation, reason = "a day of a month, 1 to 31")]
    let day = (rest - MONTH_STARTS[place] + 1) as u8;
    let year = periods * 400 + centuries * 100 + groups * 4 + years + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The instant `secs` seconds and `nanos` nanoseconds from 1970.
    fn at(secs: i64, nanos: u32) -> Timestamp {
        Timestamp { secs, nanos }
    }

    /// The calendar's two conversions undo each other on every day of the
    /// years 0 to 9999, and each day is the civil day after the one before.
    #[test]
    fn the_calendar_counts_every_day_of_the_years_0_to_9999() {
        let first = days_from_civil(0, 1, 1);
        let last = days_from_civil(9_999, 12, 31);
        assert_eq!((first, last), (-719_528, 2_932_896));
        assert_eq!(days_from_civil(1_970, 1, 1), 0);
        assert_eq!(days_from_civil(2_000, 2, 29), 11_016);
        let mut previous = civil_from_days(first - 1);
        assert_eq!(previous, (-1, 12, 31));
        for day in first..=last {
            let (year, month, date) = civil_from_days(day);
            assert_eq!(days_from_civil(year, month, date), day);
            let (y, m, d) = previous;
            let next = if d < days_in_month(y, m) {
                (y, m, d + 1)
            } else if m < 12 {
                (y, m + 1, 1)
            } else {
                (y + 1, 1, 1)
            };
            assert_eq!((year, month, date), next, "day {day}");
            previous = (year, month, date);
        }
        assert_eq!(civil_from_days(last + 1), (10_000, 1, 1));
    }

    /// A year divisible by 4 has a 29 February, but for one divisible by 100
    /// and not by 400.
    #[test]
    fn the_leap_years_are_the_gregorian_ones() {
        for year in [0, 4, 400, 1_600, 2_000, 2_024, 2_400, -4, -400] {
            assert_eq!(days_in_month(year, 2), 29, "{year}");
        }
        for year in [1, 100, 1_700, 1_900, 2_026, 2_100, -1, -100] {
            assert_eq!(days_in_month(year, 2), 28, "{year}");
        }
    }

    /// Each spelling of a set of instants, the epoch, each length of
    /// fraction and the ends of the years 0 to 9999 among them.
    #[test]
    fn each_instant_spells_as_the_engine_writes_it() {
        let day = |year, month, date| days_from_civil(year, month, date) * SECS_PER_DAY;
        let cases = [
            (
                at(0, 0),
                "1970-01-01T00:00:00+00:00",
                "19700101T000000000Z",
                "1970-01-01",
            ),
            (
                at(0, 1),
                "1970-01-01T00:00:00.000000001+00:00",
                "19700101T000000000Z",
                "1970-01-01",
            ),
            (
                at(0, 1_000),
                "1970-01-01T00:00:00.000001+00:00",
                "19700101T000000000Z",
                "1970-01-01",
            ),
            (
                at(0, 1_000_000),
                "1970-01-01T00:00:00.001+00:00",
                "19700101T000000001Z",
                "1970-01-01",
            ),
            (
                at(0, 120_000_000),
                "1970-01-01T00:00:00.120+00:00",
                "19700101T000000120Z",
                "1970-01-01",
            ),
            (
                at(0, 999_999_999),
                "1970-01-01T00:00:00.999999999+00:00",
                "19700101T000000999Z",
                "1970-01-01",
            ),
            (
                at(-1, 999_999_999),
                "1969-12-31T23:59:59.999999999+00:00",
                "19691231T235959999Z",
                "1969-12-31",
            ),
            (
                at(1_764_224_869, 767_390_800),
                "2025-11-27T06:27:49.767390800+00:00",
                "20251127T062749767Z",
                "2025-11-27",
            ),
            (
                at(day(2_000, 2, 29) + 43_200, 500_000_000),
                "2000-02-29T12:00:00.500+00:00",
                "20000229T120000500Z",
                "2000-02-29",
            ),
            (
                at(day(0, 1, 1), 0),
                "0000-01-01T00:00:00+00:00",
                "00000101T000000000Z",
                "0000-01-01",
            ),
            (
                at(day(10_000, 1, 1) - 1, 999_999_999),
                "9999-12-31T23:59:59.999999999+00:00",
                "99991231T235959999Z",
                "9999-12-31",
            ),
            (
                at(day(10_000, 1, 1), 0),
                "+10000-01-01T00:00:00+00:00",
                "+100000101T000000000Z",
                "+10000-01-01",
            ),
            (
                at(day(-1, 6, 15), 0),
                "-0001-06-15T00:00:00+00:00",
                "-00010615T000000000Z",
                "-0001-06-15",
            ),
        ];
        for (instant, rfc3339, compact, date) in cases {
            assert_eq!(instant.to_rfc3339(), rfc3339);
            assert_eq!(instant.to_compact(), compact);
            assert_eq!(instant.to_date(), date);
        }
        assert_eq!(
            at(day(10_000, 1, 1) - 1, 999_999_999).to_rfc3339().len(),
            35
        );
        assert_eq!(at(0, 0).to_compact().len(), 19);
    }

    /// What a load reads and refuses, at the edges of the set.
    #[test]
    fn a_load_reads_the_set_the_module_doc_states() {
        let epoch_noon = 1_764_246_896;
        let read = [
            ("2025-11-27T12:34:56+00:00", Some(at(epoch_noon, 0))),
            (
                "2025-11-27T12:34:56.794555200+00:00",
                Some(at(epoch_noon, 794_555_200)),
            ),
            (
                "2025-11-27T12:34:56.829229+00:00",
                Some(at(epoch_noon, 829_229_000)),
            ),
            (
                "2025-11-27T12:34:56.123+00:00",
                Some(at(epoch_noon, 123_000_000)),
            ),
            ("2025-11-27T12:34:56Z", Some(at(epoch_noon, 0))),
            ("2025-11-27t12:34:56z", Some(at(epoch_noon, 0))),
            ("2025-11-27 12:34:56Z", Some(at(epoch_noon, 0))),
            ("2025-11-27T12:34:56.1Z", Some(at(epoch_noon, 100_000_000))),
            (
                "2025-11-27T12:34:56.123456789987Z",
                Some(at(epoch_noon, 123_456_789)),
            ),
            (
                "2025-11-27T12:34:60.5Z",
                Some(at(epoch_noon + 4, 500_000_000)),
            ),
            ("2025-11-27T13:34:56+01:00", Some(at(epoch_noon, 0))),
            ("2025-11-27T07:04:56-05:30", Some(at(epoch_noon, 0))),
            ("2025-11-27T07:04:56\u{2212}05:30", Some(at(epoch_noon, 0))),
            ("2025-11-27T12:34:56-00:00", Some(at(epoch_noon, 0))),
            ("2025-11-28T12:33:56+23:59", Some(at(epoch_noon, 0))),
            ("2024-02-29T00:00:00Z", Some(at(1_709_164_800, 0))),
            ("1969-12-31T23:59:59.5Z", Some(at(-1, 500_000_000))),
            ("2025-11-27T12:34:56", None),
            ("2025-11-27x12:34:56Z", None),
            ("2025-11-27T12:34:56.Z", None),
            ("2025-11-27T12:34:56,5Z", None),
            ("2025-11-27T12:34:61Z", None),
            ("2025-11-27T24:00:00Z", None),
            ("2025-11-27T23:60:00Z", None),
            ("2025-02-29T00:00:00Z", None),
            ("1900-02-29T00:00:00Z", None),
            ("2025-13-01T00:00:00Z", None),
            ("2025-00-01T00:00:00Z", None),
            ("2025-01-00T00:00:00Z", None),
            ("2025-04-31T00:00:00Z", None),
            ("2025-1-01T00:00:00Z", None),
            ("+10000-01-01T00:00:00+00:00", None),
            ("2025-11-27T12:34:56+24:00", None),
            ("2025-11-27T12:34:56+00:60", None),
            ("2025-11-27T12:34:56+0000", None),
            ("2025-11-27T12:34:56+00", None),
            ("2025-11-27T12:34:56+00:00:00", None),
            ("2025-11-27T12:34:56\u{2013}05:00", None),
            (" 2025-11-27T12:34:56Z", None),
            ("2025-11-27T12:34:56Z ", None),
            ("2025-11-27T12:34:56+00:00\n", None),
            ("\u{FF12}025-11-27T12:34:56Z", None),
            ("", None),
            ("yesterday", None),
        ];
        for (text, instant) in read {
            assert_eq!(Timestamp::parse_rfc3339(text), instant, "{text:?}");
        }
    }

    /// splitmix64, so every run draws the same instants.
    fn draws(seed: u64) -> impl Iterator<Item = u64> {
        let mut state = seed;
        std::iter::repeat_with(move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
    }

    /// A load reads every RFC 3339 spelling back as the instant it spells,
    /// and two spellings of one kind sort as text as their instants do.
    #[test]
    fn a_spelling_reads_back_and_sorts_as_its_instant() {
        let low = days_from_civil(0, 1, 1) * SECS_PER_DAY;
        let high = days_from_civil(10_000, 1, 1) * SECS_PER_DAY - 1;
        let mut words = draws(0x0235_C10C_0000_0001);
        let mut next = || words.next().unwrap();
        let span = (high - low + 1) as u64;
        for _ in 0..100_000 {
            let a = at(
                low + (next() % span) as i64,
                (next() % 1_000_000_000) as u32,
            );
            let (secs, nanos) = match next() % 3 {
                0 => (
                    low + (next() % span) as i64,
                    (next() % 1_000_000_000) as u32,
                ),
                1 => (
                    a.secs,
                    [0, 1_000, 1_000_000, 500_000_000][(next() % 4) as usize],
                ),
                _ => (
                    a.secs + (next() % 5) as i64 - 2,
                    a.nanos / 1_000_000 * 1_000_000,
                ),
            };
            let b = at(secs.clamp(low, high), nanos);
            assert_eq!(Timestamp::parse_rfc3339(&a.to_rfc3339()), Some(a));
            let day = |t: Timestamp| t.secs.div_euclid(SECS_PER_DAY);
            let millis = |t: Timestamp| (t.secs, t.nanos / 1_000_000);
            assert_eq!(a.to_rfc3339().cmp(&b.to_rfc3339()), a.cmp(&b));
            assert_eq!(
                a.to_compact().cmp(&b.to_compact()),
                millis(a).cmp(&millis(b))
            );
            assert_eq!(a.to_date().cmp(&b.to_date()), day(a).cmp(&day(b)));
        }
    }

    /// The clock's reading lies between two readings of the system clock
    /// taken around it.
    #[test]
    fn now_is_the_system_clock() {
        let before = Timestamp::from(SystemTime::now());
        let now = Timestamp::now();
        let after = Timestamp::from(SystemTime::now());
        assert!(
            before <= now && now <= after,
            "{before:?} {now:?} {after:?}"
        );
    }

    /// A system time before 1970 is the instant it names, and spells as that
    /// instant, as a time after 1970 does. Each duration is a whole number of
    /// 100 ns, the step Windows' clock counts in, so it is the same on every
    /// system.
    #[test]
    fn a_time_before_1970_is_the_time_it_names() {
        let cases = [
            (
                Duration::new(1, 250_000_000),
                at(-2, 750_000_000),
                "1969-12-31T23:59:58.750+00:00",
                "19691231T235958750Z",
            ),
            (
                Duration::from_nanos(100),
                at(-1, 999_999_900),
                "1969-12-31T23:59:59.999999900+00:00",
                "19691231T235959999Z",
            ),
            (
                Duration::from_secs(86_400),
                at(-86_400, 0),
                "1969-12-31T00:00:00+00:00",
                "19691231T000000000Z",
            ),
            (
                Duration::new(14_182_940, 100),
                at(-14_182_941, 999_999_900),
                "1969-07-20T20:17:39.999999900+00:00",
                "19690720T201739999Z",
            ),
        ];
        for (before, instant, rfc3339, compact) in cases {
            let time = Timestamp::from(UNIX_EPOCH - before);
            assert_eq!(time, instant);
            assert_eq!(time.to_rfc3339(), rfc3339);
            assert_eq!(time.to_compact(), compact);
            assert_eq!(Timestamp::parse_rfc3339(rfc3339), Some(instant));
        }
        assert_eq!(Timestamp::from(UNIX_EPOCH), at(0, 0));
        let after = UNIX_EPOCH + Duration::new(1, 500);
        assert_eq!(Timestamp::from(after), at(1, 500));
    }
}
