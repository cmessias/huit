use crate::cpu::Cpu;
use crate::drivers::display::DisplayDriver;
use crate::drivers::input::{InputDriver, PollResult};
use std::fs::File;

pub struct Chip8 {
    cpu: Cpu,
    display: DisplayDriver,
    input: InputDriver,
}

impl Chip8 {
    pub fn new(cpu: Cpu, display: DisplayDriver, input: InputDriver) -> Chip8 {
        Chip8 {
            cpu,
            display,
            input,
        }
    }

    pub fn load(&mut self, rom: File) {
        self.cpu.load(rom);
    }

    pub fn run(&mut self) {
        loop {
            self.cpu.run_cycles(8);

            match self.input.poll() {
                PollResult::Quit => break,
                PollResult::Keys(keys) => self.cpu.press_keys(&keys),
            }

            self.cpu.tick_timers();
            self.display.draw(self.cpu.display);
        }
    }
}
