use std;
use std::io::{self, Read};
use std::marker::PhantomData;
use std::{fmt, error, convert};

use util::*;

const DEFAULT_BUFFER_SIZE : u32 = 4096; // 4KB buffer
const MIN_AVAILABLE : usize = 4; // Minimum number of words in the buffer at any given time

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    UnexpectedEOF,
    Io(io::Error)
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::UnexpectedEOF => f.write_str("Unexpected End-Of-File"),
            Error::Io(ref io) => fmt::Display::fmt(io, f)
        }
    }
}

impl convert::From<io::Error> for Error {
    fn from(io: io::Error) -> Error {
        Error::Io(io)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::UnexpectedEOF => "Unexpected End-Of-File",
            Error::Io(ref io) => io.description()
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Io(ref io) => Some(io),
            _ => None
        }
    }
}

/// `BitReader` reads an input stream as a stream of bits (instead of the normal byte-level).
pub struct BitReader<R: Read + ?Sized, E: Endianess> {
    _ph: PhantomData<*const E>,
    consumed_bits: usize,
    buf_size: u32,
    buf_len: u32,
    buffer: *mut u8,
    inner: R,
}

/// A trait for handling endian-specific operations
pub trait Endianess {
    fn swap(s: usize) -> usize;
    fn get_bits(v: usize, start: usize, len: usize) -> usize;
    fn combine_words(first: usize, second: usize) -> u64;
}

pub enum LittleEndian { }
pub enum BigEndian { }

impl Endianess for LittleEndian {
    #[inline(always)]
    fn swap(s: usize) -> usize {
        usize::from_le(s)
    }

    #[inline]
    fn get_bits(v: usize, start: usize, len: usize) -> usize {
        // Little endian source reads bits from least signficant first
        if len == word_bits() {
            debug_assert!(start == 0);
            return v;
        }
        let mask = (1 << len) - 1;
        let mask = mask << start;

        let v = v & mask;
        v >> start
    }

    fn combine_words(first: usize, second: usize) -> u64 {
        let lower = first as u64;
        lower | ((second as u64) << word_bits())
    }
}

impl Endianess for BigEndian {
    #[inline(always)]
    fn swap(s: usize) -> usize {
        usize::from_be(s)
    }

    #[inline]
    fn get_bits(v: usize, start: usize, len: usize) -> usize {
        // Big endian source reads bits from most significant first
        if len == word_bits() {
            debug_assert!(start == 0);
            return v;
        }
        let mask = (1 << len) - 1;
        let shift = word_bits() - (len + start);
        let mask = mask << shift;

        let v = v & mask;
        v >> shift
    }

    fn combine_words(first: usize, second: usize) -> u64 {
        let lower = second as u64;
        lower | ((first as u64) << word_bits())
    }
}

impl<R: Read, E:Endianess> BitReader<R, E> {
    pub fn new(rdr: R) -> BitReader<R, E> {
        BitReader::with_buffer_size(rdr, DEFAULT_BUFFER_SIZE)
    }

    pub fn with_buffer_size(rdr: R, mut sz: u32) -> BitReader<R, E> {
        unsafe {
            if sz < (MIN_AVAILABLE as u32) * (word_bytes() as u32) {
                sz = (MIN_AVAILABLE as u32) * (word_bytes() as u32);
            }
            let n = sz % (word_bytes() as u32);
            if n > 0 {
                sz += word_bytes() as u32 - n;
            }
            let buffer = allocate_buffer(sz as usize);
            BitReader {
                consumed_bits: 0,
                buf_size: sz,
                buf_len: 0,
                buffer: buffer,
                _ph: PhantomData,
                inner: rdr
            }
        }
    }
}

impl<R: Read + ?Sized, E:Endianess> BitReader<R, E> {
    /// Reads `bits` bits into a `usize`.
    /// Panics if `bits` is greater than the number of bits in a `usize`
    pub fn read_usize(&mut self, bits: u8) -> Result<usize> {
        if bits == 0 { return Ok(0); }
        let bits = bits as usize;

        assert!(bits <= word_bits());

        // Try to fill the buffer with enough bits.
        while (self.available_bytes() * 8) < bits {
            try!(self.fill_buffer());
        }

        let current_word = self.current_word();
        let consumed_bits = self.consumed_bits_in_current_word();
        let available_bits = word_bits() - consumed_bits;

        if available_bits >= bits {
            let v = E::get_bits(current_word, consumed_bits, bits);
            self.consumed_bits += bits;
            Ok(v)
        } else {
            let v = E::get_bits(current_word, consumed_bits, available_bits);
            self.consumed_bits += available_bits;
            let rest_bits = bits - available_bits;

            let current_word = self.current_word();
            let mut v = v << rest_bits;
            v |= E::get_bits(current_word, 0, rest_bits);
            Ok(v)
        }
    }

    /// Reads `bits` bits into an `isize`.
    /// Panics if `bits` is greater than the number of bits in a `isize`
    pub fn read_isize(&mut self, bits: u8) -> Result<isize> {
        if bits == 0 { return Ok(0); }
        let v = try!(self.read_usize(bits));
        let bit = 1 << (bits - 1) as usize;

        if (v & bit) == 0 {
            Ok(v as isize)
        } else {
            let ones = !0 << (bits as usize);
            let v = v | ones;
            Ok(v as isize)
        }
    }

    /// Reads `bits` bits into a `u64`.
    /// Panics if `bits` is greater than the number of bits in a `u64`
    pub fn read_u64(&mut self, bits: u8) -> Result<u64> {
        assert!(bits <= 64);
        let bits = bits as usize;
        if bits <= word_bits() {
            Ok(try!(self.read_usize(bits as u8)) as u64)
        } else {
            // Handle the case where size_of::<u64>() > size_of::<usize>
            let first = try!(self.read_usize(word_bits() as u8));
            let second = try!(self.read_usize((bits - word_bits()) as u8));
            Ok(E::combine_words(first, second))
        }
    }

    /// Reads `bits` bits into an `i64`.
    /// Panics if `bits` is greater than the number of bits in a `i64`
    pub fn read_i64(&mut self, bits: u8) -> Result<i64> {
        assert!(bits <= 64);
        let bits = bits as usize;
        if bits <= word_bits() {
            Ok(try!(self.read_isize(bits as u8)) as i64)
        } else {
            let first = try!(self.read_usize(word_bits() as u8));
            let second = try!(self.read_usize((bits - word_bits()) as u8));
            let v = E::combine_words(first, second);

            let bit = 1 << (bits - 1) as u64;
            if (v & bit) == 0 {
                Ok(v as i64)
            } else {
                let ones = !0 << (bits as u64);
                let v = v | ones;
                Ok(v as i64)
            }
        }

    }

    pub fn read_u32(&mut self, bits: u8) -> Result<u32> {
        assert!(bits <= 32);
        self.read_usize(bits).map(|v| v as u32)
            .map_err(|e| Error::from(e))
    }

    pub fn read_i32(&mut self, bits: u8) -> Result<i32> {
        assert!(bits <= 32);
        self.read_usize(bits).map(|v| v as i32)
            .map_err(|e| Error::from(e))
    }

    pub fn read_u16(&mut self, bits: u8) -> Result<u16> {
        assert!(bits <= 16);
        self.read_usize(bits).map(|v| v as u16)
            .map_err(|e| Error::from(e))
    }

    pub fn read_i16(&mut self, bits: u8) -> Result<i16> {
        assert!(bits <= 16);
        self.read_usize(bits).map(|v| v as i16)
            .map_err(|e| Error::from(e))
    }

    pub fn read_u8(&mut self, bits: u8) -> Result<u8> {
        assert!(bits <= 8);
        self.read_usize(bits).map(|v| v as u8)
            .map_err(|e| Error::from(e))
    }

    pub fn read_i8(&mut self, bits: u8) -> Result<i8> {
        assert!(bits <= 8);
        self.read_usize(bits).map(|v| v as i8)
            .map_err(|e| Error::from(e))
    }

    /// Read a single bit from the stream returning it as a boolean value
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(try!(self.read_usize(1)) != 0)
    }

    /// Read an unsigned, unary-coded value from the bitstream. This uses one-terminated encoding.
    /// For example: the number `6` is encoded as `0000001`
    pub fn read_unary_unsigned(&mut self) -> Result<usize> {
        if false { // Simple version
            let mut val = 0;
            loop {
                if !try!(self.read_bit()) {
                    val += 1;
                } else {
                    break;
                }
            }
            Ok(val)
        } else {
            let mut val = 0;
            loop {
                while self.available_words() > 1 {
                    let consumed_bits = self.consumed_bits_in_current_word();
                    let mut current_word = self.current_word();
                    current_word = E::get_bits(current_word, consumed_bits, word_bits()-consumed_bits);

                    current_word <<= consumed_bits;
                    if current_word > 0 {
                        let i = current_word.leading_zeros() as usize;
                        val += i;

                        self.consumed_bits += i+1;

                        return Ok(val);
                    } else {
                        val += word_bits() - consumed_bits;
                        self.consumed_bits += word_bits() - consumed_bits;
                    }
                }

                let bytes = self.buf_len as usize % word_bytes();
                if bytes > 0 {
                    let len = bytes * 8;
                    let consumed_bits = self.consumed_bits_in_current_word();
                    let mut current_word = self.current_word();
                    current_word = E::get_bits(current_word, consumed_bits, len);

                    if current_word > 0 {
                        let i = current_word.leading_zeros() as usize;
                        val += i;
                        self.consumed_bits += i+1;

                        return Ok(val);
                    } else {
                        val += len - consumed_bits;
                        self.consumed_bits += len-consumed_bits;
                    }
                }

                try!(self.fill_buffer());
            }
        }
    }

    /// Advances the reader such that the next bits will be from the start of a byte boundary.  If
    /// the reader is already on a byte boundary, nothing is changed.
    pub fn skip_to_byte(&mut self) {
        if (self.consumed_bits % 8) > 0 {
            self.consumed_bits += 8 - (self.consumed_bits % 8);
        }
    }

    fn current_word(&self) -> usize {
        unsafe {
            let buf = self.buffer as *mut usize;
            let current_word = self.consumed_words();
            *buf.offset(current_word as isize)
        }
    }

    fn fill_buffer(&mut self) -> Result<()> {
        let available_words = self.available_words();
        if available_words < MIN_AVAILABLE {
            self.clear_consumed();
            unsafe {
                // Get a poiner to the last valid word
                let offset = if available_words == 0 { 0 } else { (available_words - 1) as isize };
                let mut word_ptr = (self.buffer as *mut usize).offset(offset);

                // if we have a partial word at the end, we need to swap it back
                // to the source endianess. It'll get swapped back after the read.
                // Otherwise, just advance to the next word.
                if !self.buf_len_aligned() {
                    *word_ptr = E::swap(*word_ptr);
                } else if available_words > 0 {
                    word_ptr = word_ptr.offset(1);
                }

                let buf_start = self.buffer.offset(self.buf_len as isize);
                let len = self.buf_size - self.buf_len;

                let slice = std::slice::from_raw_parts_mut(buf_start, len as usize);

                // Try to fill the buffer
                let n = try!(self.inner.read(slice));

                if n == 0 {
                    return Err(Error::UnexpectedEOF);
                }

                self.buf_len += n as u32;

                // Get the pointer to just past the last available word
                let end = (self.buffer as *mut usize)
                    .offset(self.available_words() as isize);

                // Swap all the the read words to native endianess
                while word_ptr < end {
                    *word_ptr = E::swap(*word_ptr);
                    word_ptr = word_ptr.offset(1);
                }
            }

        }
        Ok(())
    }

    fn available_words(&self) -> usize {
        let consumed = self.consumed_words();
        let mut total = self.buf_len as usize / word_bytes();
        if !self.buf_len_aligned() {
            total += 1;
        }

        debug_assert!(total >= consumed);

        total - consumed
    }

    fn available_bytes(&self) -> usize {
        let total = self.buf_len as usize;
        let mut consumed = self.consumed_words() * word_bytes();
        consumed += self.consumed_bits_in_current_word() / 8;

        total - consumed
    }

    fn buf_len_aligned(&self) -> bool {
        (self.buf_len as usize % word_bytes()) == 0
    }

    fn clear_consumed(&mut self) {
        unsafe {
            let mut word_buf = self.buffer as *mut usize;
            let mut unconsumed = word_buf.offset(self.consumed_words() as isize);

            if word_buf == unconsumed { return; }

            let len = self.available_words();
            if len == 0 {
                self.buf_len = 0;
                self.consumed_bits = 0;
            }

            for _ in 0..len {
                *word_buf = *unconsumed;

                word_buf = word_buf.offset(1);
                unconsumed = unconsumed.offset(1);
            }

            let uninit = self.buf_len % (word_bytes() as u32);
            let uninit = if uninit == 0 { 0 } else { word_bytes() as u32 - uninit };

            self.buf_len = (len * word_bytes()) as u32 - uninit;
            self.consumed_bits = self.consumed_bits_in_current_word();
        }
    }

    fn consumed_words(&self) -> usize {
        self.consumed_bits / word_bits()
    }

    fn consumed_bits_in_current_word(&self) -> usize {
        self.consumed_bits % word_bits()
    }

}

impl<R:Read + ?Sized, E:Endianess> Drop for BitReader<R,E> {
    fn drop(&mut self) {
        unsafe {
            deallocate_buffer(self.buffer, self.buf_size as usize);
        }
    }
}

#[cfg(test)]
mod test {
    use std::io;
    use super::*;

    #[test]
    pub fn simple_read() {
        let r = io::Cursor::new(vec![0x0,0x1,0x2,0x3]);
        let mut stream : BitReader<_, BigEndian> = BitReader::new(r);

        let v = stream.read_usize(32).unwrap();
        assert_eq!(v, 0x00010203);

        // Make sure end-of-stream is end-of-stream
        match stream.read_usize(16) {
            Err(Error::UnexpectedEOF) => (),
            v => panic!("Expected EOF, got: {:?}", v)
        }

        // Make sure we continue to get EOFs
        match stream.read_usize(16) {
            Err(Error::UnexpectedEOF) => (),
            v => panic!("Expected EOF, got: {:?}", v)
        }
    }
}
