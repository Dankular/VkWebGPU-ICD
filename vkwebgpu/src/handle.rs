//! Handle management for Vulkan objects
//!
//! Vulkan uses opaque handles (u64 on 64-bit, dispatchable/non-dispatchable).
//! We need to map these to our internal Rust objects.
//!
//! # Dispatchable handles (VkDevice, VkQueue, VkCommandBuffer)
//!
//! The Vulkan loader (ICD interface v5) requires that the first pointer-sized
//! word of a dispatchable object is ICD_LOADER_MAGIC = 0x01CDC0DE.  The loader
//! dereferences the handle as a pointer to read this magic; if it points to
//! unmapped memory (e.g. integer 1) the process crashes with ACCESS_VIOLATION.
//!
//! Use `alloc_dispatchable(index)` to create a heap-resident DispatchableSlot
//! that satisfies this contract, and `get_dispatchable` / `remove_dispatchable`
//! on HandleAllocator to look objects up via the returned pointer handle.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Magic value the Vulkan loader expects at offset 0 of every dispatchable object.
pub const ICD_LOADER_MAGIC: usize = 0x01CDC0DE;

/// Heap-resident slot for a dispatchable Vulkan handle.
/// The loader reads `loader_magic` (first field) to verify the handle is valid.
#[repr(C)]
pub struct DispatchableSlot {
    /// MUST be first and equal to ICD_LOADER_MAGIC
    pub loader_magic: usize,
    /// Index into the corresponding HandleAllocator
    pub index: u64,
}

/// Allocate a heap-resident DispatchableSlot and return a raw pointer as u64.
/// Use this as the raw value of VkDevice / VkQueue / VkCommandBuffer handles.
pub fn alloc_dispatchable(index: u64) -> u64 {
    let slot = Box::new(DispatchableSlot {
        loader_magic: ICD_LOADER_MAGIC,
        index,
    });
    Box::into_raw(slot) as u64
}

/// Free a DispatchableSlot allocated by `alloc_dispatchable`.
/// # Safety
/// `raw` must have been produced by `alloc_dispatchable` and not freed before.
pub unsafe fn free_dispatchable(raw: u64) {
    if raw != 0 {
        drop(Box::from_raw(raw as *mut DispatchableSlot));
    }
}

/// Extract the allocator index from a dispatchable handle.
/// # Safety
/// `raw` must be a live pointer produced by `alloc_dispatchable`.
pub unsafe fn dispatchable_index(raw: u64) -> u64 {
    (*(raw as *const DispatchableSlot)).index
}

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

    /// Get an object via a *dispatchable* handle (raw pointer to DispatchableSlot).
    /// Use this for VkDevice, VkQueue, VkCommandBuffer.
    pub fn get_dispatchable(&self, raw: u64) -> Option<Arc<T>> {
        if raw == 0 {
            return None;
        }
        let index = unsafe { dispatchable_index(raw) };
        self.get(index)
    }

    /// Remove an object via a *dispatchable* handle.
    /// Does NOT free the DispatchableSlot itself; call `free_dispatchable` separately.
    pub fn remove_dispatchable(&self, raw: u64) -> Option<Arc<T>> {
        if raw == 0 {
            return None;
        }
        let index = unsafe { dispatchable_index(raw) };
        self.remove(index)
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
