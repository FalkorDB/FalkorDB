//! Choosing the fewest bytes that hold an integer.
//!
//! Two subsystems narrow integers to save space and had grown the same
//! four-armed match independently: the effects codec, which writes ids and run
//! lengths at the narrowest width, and the index B-tree's compact leaf, which
//! stores keys and doc ids the same way. Identical arms, different names and
//! return types, no shared definition — so a change to one could silently
//! disagree with the other.
//!
//! Only the *choice* is shared. Reading a narrowed value is not: the codec pulls
//! it from a bounds-checked cursor, the B-tree from a page at an offset, and
//! those have nothing in common but the width.

/// Fewest power-of-two bytes — 1, 2, 4 or 8 — that hold `max`.
///
/// Powers of two so every read is one fixed-size load rather than a
/// variable-length copy.
#[must_use]
pub const fn width_for(max: u64) -> u8 {
    match max {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFFFF_FFFF => 4,
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::width_for;

    #[test]
    fn boundaries() {
        assert_eq!(width_for(0), 1);
        assert_eq!(width_for(0xFF), 1);
        assert_eq!(width_for(0x100), 2);
        assert_eq!(width_for(0xFFFF), 2);
        assert_eq!(width_for(0x1_0000), 4);
        assert_eq!(width_for(0xFFFF_FFFF), 4);
        assert_eq!(width_for(0x1_0000_0000), 8);
        assert_eq!(width_for(u64::MAX), 8);
    }
}
