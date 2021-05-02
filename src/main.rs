use crate::chip8::Chip8;
use crate::constants::hardware::{SCREEN_HEIGHT, SCREEN_WIDTH};
use crate::drivers::display::DisplayDriver;
use crate::drivers::input::InputDriver;
use cpu::Cpu;
use sdl2::Sdl;
use std::env;
use std::fs::File;

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
    let rom_path = env::args().skip(1).next()
        .expect("Rom path must be provided as argument");
    let rom = File::open(rom_path).unwrap();
    chip8.load(rom);
    chip8.run();

    return;
}
