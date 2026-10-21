//! Tree construction macro for execution plans.
//!
//! This module provides the [`tree!`] macro for concise construction of
//! `DynTree<IR>` execution plan trees in the planner.
//!
//! ## Usage
//!
//! ```ignore
//! // Leaf node
//! tree!(IR::Return(vars))
//!
//! // Node with children
//! tree!(IR::Filter(expr), child1, child2)
//!
//! // Node with children from iterator
//! tree!(IR::Union; children_iter)
//! ```

#[macro_export]
macro_rules! tree {
    ($value:expr) => {
        DynTree::new($value)
    };
    ($value:expr, $($child:expr),*) => {
        {
            let mut n = DynTree::new($value);
            let mut root = n.root_mut();
            $(root.push_child_tree($child);)*
            n
        }
    };
    ($value:expr ; $($iter:expr),*) => {
        {
            let mut n = DynTree::new($value);
            let mut root = n.root_mut();
            $(for child in $iter {
                root.push_child_tree(child);
            })*
            n
        }
    };
    () => {};
}
