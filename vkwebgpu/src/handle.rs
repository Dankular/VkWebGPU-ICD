//! Handle management for Vulkan objects
//!
//! Vulkan uses opaque handles (u64 on 64-bit, dispatchable/non-dispatchable).
//! We need to map these to our internal Rust objects.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Thread-safe handle allocator
pub struct HandleAllocator<T> {
    next_handle: RwLock<u64>,
    objects: RwLock<FxHashMap<u64, Arc<T>>>,
}

impl<T> HandleAllocator<T> {
    pub fn new() -> Self {
        Self {
            next_handle: RwLock::new(1), // Start at 1, 0 is VK_NULL_HANDLE
            objects: RwLock::new(FxHashMap::default()),
        }
    }

    /// Allocate a new handle for an object
    pub fn allocate(&self, object: T) -> u64 {
        let mut next = self.next_handle.write();
        let handle = *next;
        *next += 1;

        let mut objects = self.objects.write();
        objects.insert(handle, Arc::new(object));

        handle
    }

    /// Get a reference to an object by handle
    pub fn get(&self, handle: u64) -> Option<Arc<T>> {
        let objects = self.objects.read();
        objects.get(&handle).cloned()
    }

    /// Remove an object by handle
    pub fn remove(&self, handle: u64) -> Option<Arc<T>> {
        let mut objects = self.objects.write();
        objects.remove(&handle)
    }

    /// Check if a handle exists
    pub fn contains(&self, handle: u64) -> bool {
        let objects = self.objects.read();
        objects.contains_key(&handle)
    }

    /// Get the count of allocated handles
    pub fn count(&self) -> usize {
        let objects = self.objects.read();
        objects.len()
    }
}

impl<T> Default for HandleAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_allocator() {
        let allocator = HandleAllocator::new();

        let handle1 = allocator.allocate(42);
        let handle2 = allocator.allocate(100);

        assert_ne!(handle1, handle2);
        assert_eq!(*allocator.get(handle1).unwrap(), 42);
        assert_eq!(*allocator.get(handle2).unwrap(), 100);

        allocator.remove(handle1);
        assert!(allocator.get(handle1).is_none());
        assert!(allocator.get(handle2).is_some());
    }

    #[test]
    fn test_handle_count() {
        let allocator = HandleAllocator::new();

        assert_eq!(allocator.count(), 0);

        let h1 = allocator.allocate("test1");
        assert_eq!(allocator.count(), 1);

        let h2 = allocator.allocate("test2");
        assert_eq!(allocator.count(), 2);

        allocator.remove(h1);
        assert_eq!(allocator.count(), 1);
    }
}
