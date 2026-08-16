//! Fixed-capacity KV-cache storage.
//!
//! Cache layout is `[kv_head, token, head_dim]`. Capacity is part of the
//! layout so append does not reallocate or move existing values.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheError(pub String);

#[derive(Debug, Clone)]
pub struct KvCache {
    key_value_heads: usize,
    head_dim: usize,
    capacity_tokens: usize,
    cached_tokens: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvCache {
    pub fn new(
        key_value_heads: usize,
        head_dim: usize,
        capacity_tokens: usize,
    ) -> Result<Self, CacheError> {
        if key_value_heads == 0 || head_dim == 0 || capacity_tokens == 0 {
            return Err(CacheError("KV-cache dimensions must be non-zero".into()));
        }
        let elements = key_value_heads
            .checked_mul(head_dim)
            .and_then(|value| value.checked_mul(capacity_tokens))
            .ok_or_else(|| CacheError("KV-cache dimensions overflow".into()))?;
        Ok(Self {
            key_value_heads,
            head_dim,
            capacity_tokens,
            cached_tokens: 0,
            keys: vec![0.0; elements],
            values: vec![0.0; elements],
        })
    }

    pub fn key_value_heads(&self) -> usize {
        self.key_value_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.capacity_tokens - self.cached_tokens
    }

    pub fn key_storage(&self) -> &[f32] {
        &self.keys
    }

    pub fn value_storage(&self) -> &[f32] {
        &self.values
    }

    pub fn active_bytes(&self) -> u64 {
        (self.key_value_heads * self.cached_tokens * self.head_dim * 2 * std::mem::size_of::<f32>())
            as u64
    }

    pub fn append_token(&mut self, keys: &[f32], values: &[f32]) -> Result<(), CacheError> {
        let expected = self.key_value_heads * self.head_dim;
        if keys.len() != expected || values.len() != expected {
            return Err(CacheError(format!(
                "KV token must contain {expected} values per key and value"
            )));
        }
        if self.cached_tokens == self.capacity_tokens {
            return Err(CacheError("KV-cache capacity exhausted".into()));
        }
        for head in 0..self.key_value_heads {
            let source_start = head * self.head_dim;
            let destination_start =
                (head * self.capacity_tokens + self.cached_tokens) * self.head_dim;
            let source_end = source_start + self.head_dim;
            let destination_end = destination_start + self.head_dim;
            self.keys[destination_start..destination_end]
                .copy_from_slice(&keys[source_start..source_end]);
            self.values[destination_start..destination_end]
                .copy_from_slice(&values[source_start..source_end]);
        }
        self.cached_tokens += 1;
        Ok(())
    }

    /// Append token-major prefill data: `[token, kv_head, head_dim]`.
    pub fn append_sequence(
        &mut self,
        keys: &[f32],
        values: &[f32],
        tokens: usize,
    ) -> Result<(), CacheError> {
        let per_token = self.key_value_heads * self.head_dim;
        let expected = tokens
            .checked_mul(per_token)
            .ok_or_else(|| CacheError("KV sequence dimensions overflow".into()))?;
        if keys.len() != expected || values.len() != expected {
            return Err(CacheError(format!(
                "KV sequence must contain {expected} values per key and value"
            )));
        }
        if tokens > self.remaining_tokens() {
            return Err(CacheError(
                "KV-cache sequence exceeds remaining capacity".into(),
            ));
        }
        for token in 0..tokens {
            let start = token * per_token;
            let end = start + per_token;
            self.append_token(&keys[start..end], &values[start..end])?;
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.cached_tokens = 0;
    }

    /// Discard a speculative suffix while retaining the fixed-capacity backing
    /// storage. This is used by exact batched verification when only a prefix
    /// of the candidate sequence agrees with the target model.
    pub fn truncate_to(&mut self, cached_tokens: usize) -> Result<(), CacheError> {
        if cached_tokens > self.cached_tokens {
            return Err(CacheError(
                "cannot truncate KV-cache to a future position".into(),
            ));
        }
        self.cached_tokens = cached_tokens;
        Ok(())
    }

    /// Copy only the active prefix from another cache with matching head shape.
    /// The destination may have a larger capacity than the source. This lets
    /// branch-local speculative caches use a compact `prefix + depth` backing
    /// allocation while committing into the full target cache.
    pub fn copy_prefix_from(&mut self, source: &Self) -> Result<(), CacheError> {
        if self.key_value_heads != source.key_value_heads
            || self.head_dim != source.head_dim
            || self.capacity_tokens < source.cached_tokens
        {
            return Err(CacheError("KV-cache layouts do not match".into()));
        }
        let active_elements = source.cached_tokens * self.head_dim;
        for head in 0..self.key_value_heads {
            let source_start = head * source.capacity_tokens * source.head_dim;
            let destination_start = head * self.capacity_tokens * self.head_dim;
            let source_end = source_start + active_elements;
            let destination_end = destination_start + active_elements;
            self.keys[destination_start..destination_end]
                .copy_from_slice(&source.keys[source_start..source_end]);
            self.values[destination_start..destination_end]
                .copy_from_slice(&source.values[source_start..source_end]);
        }
        self.cached_tokens = source.cached_tokens;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_token_major_prefill_into_head_major_capacity_layout() {
        let mut cache = KvCache::new(2, 2, 3).expect("valid cache");
        let keys = [
            1.0_f32, 2.0, 3.0, 4.0, // token 0, heads 0/1
            5.0, 6.0, 7.0, 8.0, // token 1, heads 0/1
        ];
        let values = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        cache
            .append_sequence(&keys, &values, 2)
            .expect("prefill fits");
        assert_eq!(cache.cached_tokens(), 2);
        assert_eq!(cache.remaining_tokens(), 1);
        assert_eq!(cache.active_bytes(), 64);
        assert_eq!(
            cache.key_storage(),
            &[1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 3.0, 4.0, 7.0, 8.0, 0.0, 0.0]
        );
        assert_eq!(cache.value_storage()[0..4], [10.0, 20.0, 50.0, 60.0]);
        assert_eq!(cache.value_storage()[6..10], [30.0, 40.0, 70.0, 80.0]);
    }

    #[test]
    fn rejects_bad_token_shape_and_capacity_overflow() {
        let mut cache = KvCache::new(1, 2, 1).expect("valid cache");
        assert!(cache.append_token(&[1.0], &[2.0, 3.0]).is_err());
        cache
            .append_token(&[1.0, 2.0], &[3.0, 4.0])
            .expect("first token fits");
        assert!(cache.append_token(&[5.0, 6.0], &[7.0, 8.0]).is_err());
    }

    #[test]
    fn clear_reuses_allocated_capacity() {
        let mut cache = KvCache::new(2, 4, 8).expect("valid cache");
        let pointer = cache.key_storage().as_ptr();
        cache
            .append_token(&[0.0; 8], &[0.0; 8])
            .expect("token fits");
        cache.clear();
        assert_eq!(cache.cached_tokens(), 0);
        assert_eq!(cache.key_storage().as_ptr(), pointer);
    }

    #[test]
    fn truncate_to_discards_speculative_suffix_without_reallocating() {
        let mut cache = KvCache::new(1, 2, 4).expect("valid cache");
        cache
            .append_sequence(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2)
            .expect("sequence fits");
        let key_pointer = cache.key_storage().as_ptr();
        let value_pointer = cache.value_storage().as_ptr();

        cache.truncate_to(1).expect("truncation fits");

        assert_eq!(cache.cached_tokens(), 1);
        assert_eq!(cache.key_storage().as_ptr(), key_pointer);
        assert_eq!(cache.value_storage().as_ptr(), value_pointer);
        assert_eq!(cache.key_storage()[..2], [1.0, 2.0]);
        assert_eq!(cache.value_storage()[..2], [5.0, 6.0]);
    }

    #[test]
    fn truncate_to_rejects_future_position() {
        let mut cache = KvCache::new(1, 2, 2).expect("valid cache");
        cache
            .append_token(&[1.0, 2.0], &[3.0, 4.0])
            .expect("token fits");
        assert!(cache.truncate_to(2).is_err());
    }

    #[test]
    fn copy_prefix_reuses_destination_storage_and_discards_old_suffix() {
        let mut source = KvCache::new(1, 2, 4).expect("valid source cache");
        source
            .append_sequence(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2)
            .expect("source sequence fits");
        let mut destination = KvCache::new(1, 2, 4).expect("valid destination cache");
        destination
            .append_sequence(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0], 2)
            .expect("destination sequence fits");
        let key_pointer = destination.key_storage().as_ptr();
        destination
            .copy_prefix_from(&source)
            .expect("matching layouts should copy");
        assert_eq!(destination.key_storage().as_ptr(), key_pointer);
        assert_eq!(destination.cached_tokens(), 2);
        assert_eq!(destination.key_storage()[..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(destination.value_storage()[..4], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn copy_prefix_allows_compact_source_into_larger_destination() {
        let mut source = KvCache::new(2, 2, 3).expect("valid compact source");
        source
            .append_sequence(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                2,
            )
            .expect("source sequence fits");
        let mut destination = KvCache::new(2, 2, 8).expect("valid destination");
        destination
            .copy_prefix_from(&source)
            .expect("larger destination should accept active source prefix");
        assert_eq!(destination.cached_tokens(), 2);
        assert_eq!(destination.key_storage()[..4], [1.0, 2.0, 5.0, 6.0]);
        assert_eq!(destination.key_storage()[16..20], [3.0, 4.0, 7.0, 8.0]);
        assert_eq!(destination.value_storage()[..4], [11.0, 12.0, 15.0, 16.0]);
        assert_eq!(
            destination.value_storage()[16..20],
            [13.0, 14.0, 17.0, 18.0]
        );
    }
}
