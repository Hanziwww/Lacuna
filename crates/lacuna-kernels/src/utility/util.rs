pub const STRIPE: usize = 8192;
pub const STRIPE_ROWS: usize = 8192;
pub const SMALL_DIM_LIMIT: usize = 2048;
pub const SMALL_NNZ_LIMIT: usize = 32 * 1024;

pub type DenseStripe = (Vec<f64>, Vec<u8>, Vec<usize>);
pub type StripeAccs = Vec<Option<DenseStripe>>;

#[inline]
pub fn i64_to_usize(x: i64) -> usize {
    debug_assert!(x >= 0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        x as usize
    }
}

pub struct UsizeF64Map {
    keys: Vec<usize>,
    vals: Vec<f64>,
    mask: usize,
    len: usize,
}

impl UsizeF64Map {
    #[inline]
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

    #[inline]
    const fn hash(key: usize) -> usize {
        key.wrapping_mul(0x9E37_79B9_7F4A_7C15usize)
    }

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

