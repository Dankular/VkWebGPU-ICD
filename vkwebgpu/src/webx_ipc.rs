//! WebX x86 I/O port IPC transport
//!
//! Implements the same WEBX wire protocol as `guest-icd/src/vkwebx.c`, but in Rust.
//! Used exclusively when the `webx` feature is active (x86-64 Linux guest in CheerpX).
//!
//! ## Protocol
//!
//! Command packet (guest → host, via outb() byte-by-byte):
//!   [magic: u32 LE][cmd: u32 LE][seq: u32 LE][len: u32 LE][payload: len bytes]
//!
//! Response packet (host → guest, via inl()/inb()):
//!   [seq: u32 LE]  ← poll inl() until != 0xFFFFFFFF
//!   [result: i32 LE][len: u32 LE][data: len bytes]  ← read via inb()
//!
//! CheerpX returns 0xFFFFFFFF from inl() when the receive FIFO is empty.
//! The guest polls until it gets the expected seq number.

use crate::error::{Result, VkError};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU32, Ordering};

const WEBX_MAGIC: u32 = 0x58574756; // "VGWX" in little-endian

// SYS_iopl / SYS_ioperm syscall numbers for x86_64 Linux
const SYS_IOPL:   u64 = 172;
const SYS_IOPERM: u64 = 173;

// Sentinel returned by inl() when the CheerpX FIFO is empty
const FIFO_EMPTY: u32 = 0xFFFF_FFFF;

static WEBX_IPC: Lazy<WebXIpc> = Lazy::new(|| {
    // Read port from env (set by cheerpx-host.mjs as WEBX_PORT=0x7860)
    let port = std::env::var("WEBX_PORT")
        .ok()
        .and_then(|s| {
            let s = s.trim();
            let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
            u16::from_str_radix(s, 16).ok()
        })
        .unwrap_or(0x7860);

    WebXIpc::new(port)
});

/// Global singleton for the x86 I/O port IPC channel.
pub struct WebXIpc {
    port: u16,
    seq:  AtomicU32,
}

impl WebXIpc {
    /// Returns the global IPC instance (initialises on first call).
    pub fn global() -> &'static WebXIpc {
        &WEBX_IPC
    }

    fn new(port: u16) -> Self {
        let ipc = WebXIpc { port, seq: AtomicU32::new(1) };
        ipc.port_init();
        ipc
    }

    /// Grant the process I/O port access via ioperm(3), falling back to iopl(3).
    fn port_init(&self) {
        let port = self.port as u64;
        unsafe {
            // Try ioperm first — grants access to exactly 4 ports (one u32).
            let ret: i64;
            core::arch::asm!(
                "syscall",
                in("rax")  SYS_IOPERM,
                in("rdi")  port,
                in("rsi")  4u64,   // number of ports
                in("rdx")  1u64,   // turn_on = 1
                lateout("rax") ret,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
            if ret == 0 {
                log::debug!("[WebXIpc] ioperm(0x{:x}, 4, 1) succeeded", port);
                return;
            }
            log::warn!("[WebXIpc] ioperm failed ({}), trying iopl(3)", ret);

            // Fallback: iopl(3) — grants full I/O port access for the process.
            let ret2: i64;
            core::arch::asm!(
                "syscall",
                in("rax")  SYS_IOPL,
                in("rdi")  3u64,   // level 3 = full I/O access
                lateout("rax") ret2,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
            if ret2 != 0 {
                log::error!("[WebXIpc] iopl(3) also failed ({}). \
                             I/O port access unavailable — IPC will not work.", ret2);
            } else {
                log::debug!("[WebXIpc] iopl(3) succeeded");
            }
        }
    }

    // ── Allocate a sequence number (skip the sentinel value) ─────────────────

    fn next_seq(&self) -> u32 {
        loop {
            let s = self.seq.fetch_add(1, Ordering::Relaxed);
            if s != FIFO_EMPTY {
                return s;
            }
        }
    }

    // ── Low-level I/O port primitives ─────────────────────────────────────────

    #[inline(always)]
    unsafe fn outb(&self, byte: u8) {
        core::arch::asm!(
            "out dx, al",
            in("dx") self.port,
            in("al") byte,
            options(nomem, nostack, preserves_flags),
        );
    }

    #[inline(always)]
    unsafe fn inb(&self) -> u8 {
        let byte: u8;
        core::arch::asm!(
            "in al, dx",
            in("dx") self.port,
            out("al") byte,
            options(nomem, nostack, preserves_flags),
        );
        byte
    }

    #[inline(always)]
    unsafe fn inl(&self) -> u32 {
        let val: u32;
        core::arch::asm!(
            "in eax, dx",
            in("dx")   self.port,
            out("eax") val,
            options(nomem, nostack, preserves_flags),
        );
        val
    }

    // ── High-level packet send/receive ────────────────────────────────────────

    /// Send `cmd` with `payload`, block until the host response arrives.
    /// Returns the response data bytes on success or a VkError on failure.
    pub fn send_cmd(&self, cmd: u32, payload: &[u8]) -> Result<Vec<u8>> {
        let seq = self.next_seq();
        let len = payload.len() as u32;

        // Build 16-byte header
        let mut header = [0u8; 16];
        header[0..4].copy_from_slice(&WEBX_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&cmd.to_le_bytes());
        header[8..12].copy_from_slice(&seq.to_le_bytes());
        header[12..16].copy_from_slice(&len.to_le_bytes());

        // Send header + payload byte-by-byte via outb
        unsafe {
            for &b in &header  { self.outb(b); }
            for &b in payload  { self.outb(b); }
        }

        // Read response ───────────────────────────────────────────────────────
        //
        // The host sends back: [seq: u32][result: i32][data_len: u32][data: ...]
        //
        // CheerpX delivers the full Uint8Array atomically but we read it
        // byte-by-byte.  inl() reads 4 bytes at once; once the first 4 bytes
        // (the seq echo) are available, inl() returns them as a u32.
        // The remaining bytes are read with inb().

        let resp_seq = self.poll_u32();
        if resp_seq != seq {
            return Err(VkError::DeviceCreationFailed(format!(
                "WebXIpc: seq mismatch — expected {}, got {}",
                seq, resp_seq
            )));
        }

        let result_bytes = self.read_n::<4>();
        let len_bytes    = self.read_n::<4>();
        let result   = i32::from_le_bytes(result_bytes);
        let data_len = u32::from_le_bytes(len_bytes) as usize;

        let data = self.read_vec(data_len);

        if result < 0 {
            return Err(VkError::DeviceCreationFailed(format!(
                "WebXIpc: host VkResult = {}",
                result
            )));
        }

        Ok(data)
    }

    /// Poll inl() until the first 4 bytes of the response are available
    /// (i.e. until the value is not the FIFO_EMPTY sentinel).
    fn poll_u32(&self) -> u32 {
        unsafe {
            loop {
                let v = self.inl();
                if v != FIFO_EMPTY {
                    return v;
                }
                core::hint::spin_loop();
            }
        }
    }

    /// Read exactly N bytes via inb(), returning them as a fixed-size array.
    fn read_n<const N: usize>(&self) -> [u8; N] {
        let mut buf = [0u8; N];
        unsafe {
            for b in &mut buf { *b = self.inb(); }
        }
        buf
    }

    /// Read exactly `n` bytes via inb(), returning them as a Vec.
    fn read_vec(&self, n: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(n);
        unsafe {
            for _ in 0..n { buf.push(self.inb()); }
        }
        buf
    }
}
