use std::fs::File;
use std::time::{Duration, Instant};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::Sdl;

use constants::hardware::SCALE_FACTOR;
use constants::hardware::{SCREEN_HEIGHT, SCREEN_WIDTH};
use constants::instruction::*;
use cpu::Cpu;
use cpu::Pixel;

use crate::drivers::display::DisplayDriver;
use crate::drivers::input::InputDriver;
use crate::drivers::input::PollResult;

mod chip8;
mod constants;
mod cpu;
mod drivers;

fn main() {
    let sdl_context: Sdl = sdl2::init().unwrap();

    let screen_width = (SCREEN_WIDTH * SCALE_FACTOR) as u32;
    let screen_height = (SCREEN_HEIGHT * SCALE_FACTOR) as u32;

    let mut display = DisplayDriver::new(&sdl_context, screen_width, screen_height);
    let mut input = InputDriver::new(&sdl_context);
    let mut cpu = Cpu::new();
    let mut f = File::open("roms/tetris.ch8").unwrap();
    cpu.load(f);

    loop {
        cpu.run_cycles(8);

        match input.poll() {
            PollResult::Quit => break,
            PollResult::Keys(keys) => cpu.press_keys(&keys),
        }

        cpu.tick_timers();
        display.draw(cpu.display);
    }

    return;
}
