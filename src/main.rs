mod constants;
mod cpu;

use constants::instruction::*;

use constants::hardware::{MEMORY_SIZE, ENTRY_POINT, SCREEN_HEIGHT, SCREEN_WIDTH, STACK_SIZE};
use std::collections::{VecDeque, HashMap};

use cpu::Cpu;
use std::fs::File;
use std::io::{Read, BufReader};

fn main() {
    let mut cpu = Cpu::new();
    let mut f = File::open("roms/ibm_logo.ch8").unwrap();
    cpu.load(f);
    cpu.run();
}
