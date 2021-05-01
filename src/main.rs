use std::fs::File;

use sdl2::Sdl;

use cpu::Cpu;

use crate::chip8::Chip8;
use crate::constants::hardware::{SCREEN_HEIGHT, SCREEN_WIDTH};
use crate::drivers::display::DisplayDriver;
use crate::drivers::input::InputDriver;

mod chip8;
mod constants;
mod cpu;
mod drivers;

fn main() {
    let sdl_context: Sdl = sdl2::init().unwrap();

    let display = DisplayDriver::new(&sdl_context, SCREEN_WIDTH, SCREEN_HEIGHT);
    let input = InputDriver::new(&sdl_context);
    let cpu = Cpu::new();

    let mut chip8 = Chip8::new(cpu, display, input);
    let rom = File::open("roms/tetris.ch8").unwrap();
    chip8.load(rom);
    chip8.run();

    return;
}
