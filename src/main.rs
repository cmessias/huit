use std::fs::File;
use std::time::Duration;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::Sdl;

use constants::hardware::{SCREEN_HEIGHT, SCREEN_WIDTH};
use constants::hardware::SCALE_FACTOR;
use constants::instruction::*;
use cpu::Cpu;
use cpu::Pixel;
use crate::drivers::display::DisplayDriver;

mod constants;
mod cpu;
mod drivers;
mod chip8;

fn main() {
    let mut cpu = Cpu::new();
    let mut f = File::open("roms/ibm_logo.ch8").unwrap();
    cpu.load(f);
    cpu.run();

    // sdl stuff
    let sdl_context: Sdl = sdl2::init().unwrap();

    let screen_width = (SCREEN_WIDTH * SCALE_FACTOR) as u32;
    let screen_height = (SCREEN_HEIGHT * SCALE_FACTOR) as u32;
    let mut display = DisplayDriver::new(&sdl_context, screen_width, screen_height);
    display.draw(cpu.display);


    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running;
                }
                _ => {}
            }
        }

        std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}
