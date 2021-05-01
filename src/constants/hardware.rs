pub const MEMORY_SIZE: usize = 4096;
pub const ENTRY_POINT: usize = 0x200;

pub const INTERNAL_SCREEN_WIDTH: usize = 64;
pub const INTERNAL_SCREEN_HEIGHT: usize = 32;
pub const SCREEN_SIZE: usize = INTERNAL_SCREEN_WIDTH * INTERNAL_SCREEN_HEIGHT;

// Factor to increase the size of each chip-8 pixel.
// Does not increase the amount of pixels on screen, just the size of each.
pub const SCALE_FACTOR: usize = 20;

pub const SCREEN_WIDTH: u32 = (INTERNAL_SCREEN_WIDTH * SCALE_FACTOR) as u32;
pub const SCREEN_HEIGHT: u32 = (INTERNAL_SCREEN_HEIGHT * SCALE_FACTOR) as u32;

pub const FONT_ENTRY_POINT: usize = 0x50;

pub const STACK_SIZE: usize = 16;
