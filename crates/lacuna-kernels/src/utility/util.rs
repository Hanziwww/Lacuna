//! Utility constants, types, and hash map for sparse matrix operations
//
// This module provides constants for block sizes, type aliases for parallel accumulators,
// and a custom hash map for accumulating values by index. Used throughout kernels for performance.

/// Stripe/block size for parallel reductions and accumulators
pub const STRIPE: usize = 8192;
/// Stripe/block size for row-wise operations
pub const STRIPE_ROWS: usize = 8192;
/// Threshold for switching between dense and sparse algorithms (dimension)
pub const SMALL_DIM_LIMIT: usize = 2048;
/// Threshold for switching between dense and sparse algorithms (nnz)
pub const SMALL_NNZ_LIMIT: usize = 32 * 1024;

/// Type alias for a dense stripe accumulator: (values, seen mask, touched indices)
pub type DenseStripe = (Vec<f64>, Vec<u8>, Vec<usize>);
/// Type alias for a vector of optional dense stripe accumulators
pub type StripeAccs = Vec<Option<DenseStripe>>;

/// Convert i64 to usize, asserting non-negativity.
#[inline]
#[must_use]
pub fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

/// A custom hash map for accumulating f64 values by usize index.
/// Used for parallel reductions in sparse matrix kernels.
pub struct UsizeF64Map {
    keys: Vec<usize>,
    vals: Vec<f64>,
    mask: usize,
    len: usize,
}

impl UsizeF64Map {
    /// Create a new map with at least the given capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let cap2 = cap.next_power_of_two().max(16);
        let keys = vec![usize::MAX; cap2];
        let vals = vec![0.0f64; cap2];
        let mask = cap2 - 1;
        Self {
            keys,
            vals,
            mask,
            len: 0,
        }
    }

    /// Hash function for keys (multiplicative, avalanche)
    #[inline]
    const fn hash(key: usize) -> usize {
        key.wrapping_mul(0x9E37_79B9_7F4A_7C15usize)
    }

    /// Insert a value for a key, adding to any existing value.
    /// Grows the map if load factor exceeds 0.7.
    #[inline]
    pub fn insert_add(&mut self, key: usize, val: f64) {
        debug_assert!(key != usize::MAX);
        let mut idx = Self::hash(key) & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == usize::MAX {
                unsafe {
                    *self.keys.get_unchecked_mut(idx) = key;
                    *self.vals.get_unchecked_mut(idx) = val;
                }
                self.len += 1;
                if self.len * 10 > self.keys.len() * 7 {
                    self.grow();
                }
                return;
            }
            if k == key {
                unsafe {
                    *self.vals.get_unchecked_mut(idx) += val;
                }
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Grow the map to double its previous capacity, rehashing all entries.
    #[inline]
    fn grow(&mut self) {
        let new_cap = self.keys.len() * 2;
        let mut new_keys = vec![usize::MAX; new_cap];
        let mut new_vals = vec![0.0f64; new_cap];
        let new_mask = new_cap - 1;
        let old_len = self.keys.len();
        for i in 0..old_len {
            let k = self.keys[i];
            if k != usize::MAX {
                let mut idx = Self::hash(k) & new_mask;
                loop {
                    if new_keys[idx] == usize::MAX {
                        new_keys[idx] = k;
                        new_vals[idx] = self.vals[i];
                        break;
                    }
                    idx = (idx + 1) & new_mask;
                }
            }
        }
        self.keys = new_keys;
        self.vals = new_vals;
        self.mask = new_mask;
    }

    /// Drain all entries into the provided output slice, adding values by key.
    #[inline]
    pub fn drain_to(&mut self, out: &mut [f64]) {
        let len = self.keys.len();
        for i in 0..len {
            let k = self.keys[i];
            if k != usize::MAX {
                unsafe {
                    let p = out.get_unchecked_mut(k);
                    *p += self.vals[i];
                }
            }
        }
    }

    /// Return all (key, value) pairs in the map as a vector.
    #[inline]
    #[must_use]
    pub fn pairs(&self) -> Vec<(usize, f64)> {
        let len = self.keys.len();
        let mut out = Vec::with_capacity(self.len);
        for i in 0..len {
            let k = self.keys[i];
            if k != usize::MAX {
                out.push((k, self.vals[i]));
            }
        }
        out
    }
}
