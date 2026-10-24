use redis_module::RedisString;
use std::sync::Arc;

/// Owns the Redis key across async dispatch; display text never identifies a key.
#[derive(Clone)]
pub struct GraphKey {
    display: Arc<str>,
    binary: Option<Arc<[u8]>>,
}

impl GraphKey {
    pub fn new(key: &RedisString) -> Self {
        let bytes = key.as_slice();
        match std::str::from_utf8(bytes) {
            Ok(name) => Self {
                display: Arc::from(name),
                binary: None,
            },
            Err(_) => Self {
                display: Arc::from(key.to_string_lossy()),
                binary: Some(Arc::from(bytes)),
            },
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.binary.as_deref().unwrap_or(self.display.as_bytes())
    }

    pub fn display_name(&self) -> &Arc<str> {
        &self.display
    }
}
