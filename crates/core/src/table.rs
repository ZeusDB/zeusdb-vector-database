//! What a table of the standard library's `HashMap` holds, for the memory a
//! structure prices.
//!
//! The standard library's map is hashbrown's. It holds its entries in a table
//! of a power of two of buckets, at least four, and sizes the table only where
//! an insert finds no room or a caller reserves or shrinks. A removal never
//! changes the table. Right after the table is sized, the map reports seven
//! eighths of its buckets as its capacity, or one less than the buckets below
//! eight. A removal that cannot free its slot for reuse leaves a tombstone,
//! which the capacity stops counting, and whether it can depends on where the
//! map's random seed placed its neighbours. So the capacity read after a
//! removal is not the table the map holds.
//!
//! [`SizedMap`] keeps the table's buckets beside the map, taken at each call
//! that can size the table, so a price taken from them is the table the map
//! holds, with the slots removals leave. hashbrown chooses between growing a
//! table and rehashing it in place by the room its tombstones leave, so two
//! maps with one history of removals and inserts can hold tables of two sizes,
//! and each is priced at the table it holds.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;

/// Control bytes a table allocates past its last bucket, being the width of
/// the group hashbrown reads its control bytes in. The group is sixteen bytes
/// on x86 targets with SSE2, and eight on aarch64 with NEON and on the targets
/// that take hashbrown's generic group. Every target prices sixteen.
pub const HASH_CONTROL_TAIL: usize = 16;

/// The buckets of a table that reports `capacity` right after it was sized.
///
/// Undoes the seven eighths and rounds up to the power of two, at the floor
/// of four buckets an allocated table holds. A table that has seen a removal
/// since it was sized can report less than its buckets hold, so a caller reads
/// this where the table was just sized and a removal has not reached it.
pub fn table_buckets(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    (capacity.saturating_mul(8) / 7).next_power_of_two().max(4)
}

/// The capacity a table of `buckets` reports right after it was sized, which
/// is what the table holds once it is emptied in place.
pub fn bucket_capacity(buckets: usize) -> usize {
    if buckets < 8 {
        buckets.saturating_sub(1)
    } else {
        buckets / 8 * 7
    }
}

/// Bytes a table of `buckets` asks the allocator for, being the bucket array
/// of `(K, V)`, a control byte a bucket and the control tail. Nothing where
/// the map holds no table.
pub fn table_bytes_of<K, V>(buckets: usize) -> usize {
    if buckets == 0 {
        return 0;
    }
    buckets * std::mem::size_of::<(K, V)>() + buckets + HASH_CONTROL_TAIL
}

/// A `HashMap` with the buckets of its table kept beside it.
///
/// Every call here that can size the table takes the buckets again, and a
/// removal leaves them. The map is read through `Deref` and written through
/// these calls alone, so no call sizes the table unseen. The table grows and
/// rehashes as the standard map's does.
#[derive(Clone, Debug)]
pub struct SizedMap<K, V> {
    map: HashMap<K, V>,
    buckets: usize,
}

impl<K, V> Default for SizedMap<K, V> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            buckets: 0,
        }
    }
}

impl<K: Eq + Hash, V> SizedMap<K, V> {
    /// An empty map, which holds no table.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty map whose table holds `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        let map = HashMap::with_capacity(capacity);
        let buckets = table_buckets(map.capacity());
        Self { map, buckets }
    }

    /// A map the caller built, which no removal has reached since its table
    /// was last sized, so its capacity counts every slot of its table.
    pub fn from_unremoved(map: HashMap<K, V>) -> Self {
        let buckets = table_buckets(map.capacity());
        Self { map, buckets }
    }

    /// Insert `value` under `key`, and take the table's buckets again. An
    /// insert that sized the table reports the new table's whole capacity,
    /// and one that did not reports no more than its table holds, so the
    /// larger of the two counts is the table's.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let replaced = self.map.insert(key, value);
        self.buckets = self.buckets.max(table_buckets(self.map.capacity()));
        replaced
    }

    /// Remove `key`, which leaves the table and its buckets as they are.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(key)
    }
}

impl<K, V> SizedMap<K, V> {
    /// The buckets of the table the map holds.
    pub fn buckets(&self) -> usize {
        self.buckets
    }

    /// The capacity the table reports once it is emptied in place.
    pub fn table_capacity(&self) -> usize {
        bucket_capacity(self.buckets)
    }

    /// Bytes the table asks the allocator for, with the slots removals leave.
    pub fn table_bytes(&self) -> usize {
        table_bytes_of::<K, V>(self.buckets)
    }
}

impl<K, V> Deref for SizedMap<K, V> {
    type Target = HashMap<K, V>;

    fn deref(&self) -> &HashMap<K, V> {
        &self.map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table sized for the capacity a table of each count of buckets
    /// reports holds that count, and reports that capacity, so the two
    /// functions answer each other.
    #[test]
    fn a_sized_tables_buckets_and_capacity_answer_each_other() {
        assert_eq!((table_buckets(0), bucket_capacity(0)), (0, 0));
        for buckets in (2..=14).map(|power| 1usize << power) {
            let map: HashMap<u64, u64> = HashMap::with_capacity(bucket_capacity(buckets));
            assert_eq!(
                map.capacity(),
                bucket_capacity(buckets),
                "{buckets} buckets"
            );
            assert_eq!(table_buckets(map.capacity()), buckets, "{buckets} buckets");
        }
    }

    /// A map keeps its table's buckets through every removal and takes the
    /// larger count when an insert grows the table, whatever its seed. The
    /// count is the one a map given the same keys and no removal reports.
    #[test]
    fn a_map_keeps_its_tables_buckets_through_removals_and_growth() {
        for round in 0..8 {
            let mut map: SizedMap<u64, u64> = SizedMap::new();
            assert_eq!((map.buckets(), map.table_bytes()), (0, 0), "round {round}");
            let mut unremoved: HashMap<u64, u64> = HashMap::new();
            for key in 0..1_792u64 {
                map.insert(key, key);
                unremoved.insert(key, key);
            }
            assert_eq!(map.buckets(), 2_048, "round {round}");
            assert_eq!(
                map.buckets(),
                table_buckets(unremoved.capacity()),
                "round {round}"
            );
            let bytes = map.table_bytes();
            assert_eq!(bytes, 2_048 * (16 + 1) + HASH_CONTROL_TAIL, "round {round}");
            for key in 0..1_780u64 {
                assert_eq!(map.remove(&key), Some(key), "round {round}");
            }
            assert_eq!(map.len(), 12, "round {round}");
            assert_eq!(
                map.table_bytes(),
                bytes,
                "round {round}: the removals leave the table its 2,048 buckets"
            );
            // Past the whole capacity of the table, so it grows once whatever
            // its tombstones, to the 4,096 buckets that hold 3,584 entries.
            for key in 2_000..3_800u64 {
                map.insert(key, key);
            }
            assert_eq!(map.len(), 1_812, "round {round}");
            assert_eq!(map.buckets(), 4_096, "round {round}");
            assert_eq!(map.table_capacity(), 3_584, "round {round}");
        }
    }
}
