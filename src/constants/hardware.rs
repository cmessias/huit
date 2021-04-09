pub const MEMORY_SIZE: usize = 4096;
pub const ENTRY_POINT: usize = 0x200;

pub const SCREEN_WIDTH: usize = 64;
pub const SCREEN_HEIGHT: usize = 32;
pub const SCREEN_SIZE: usize = SCREEN_WIDTH * SCREEN_HEIGHT;

// Factor to increase the size of each chip-8 pixel.
// Does not increase the amount of pixels on screen, just the size of each.
pub const SCALE_FACTOR: usize = 20;

pub const FONT_ENTRY_POINT: usize = 0x50;

pub const STACK_SIZE: usize = 16;