#![feature(alloc, heap_api)]

extern crate alloc;

pub mod reader;

pub use reader::{BitReader,BigEndian,LittleEndian};

mod util {
    use std;
    // Allocates a byte buffer with word alignment
    pub unsafe fn allocate_buffer(size: usize) -> *mut u8 {
        use alloc::heap::allocate;
        debug_assert!(size % word_bytes() == 0);
        allocate(size, std::mem::align_of::<usize>())
    }

    pub unsafe fn deallocate_buffer(buf: *mut u8, size: usize) {
        use alloc::heap::deallocate;
        deallocate(buf, size, std::mem::align_of::<usize>());
    }

    #[inline(always)]
    pub fn word_bytes() -> usize {
        std::mem::size_of::<usize>()
    }
    #[inline(always)]
    pub fn word_bits() -> usize {
        std::mem::size_of::<usize>() * 8
    }
}
