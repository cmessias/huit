mod constants;
mod cpu;

use constants::instruction::*;

use constants::hardware::{MEMORY_SIZE, ENTRY_POINT, SCREEN_HEIGHT, SCREEN_WIDTH, STACK_SIZE};
use std::collections::{VecDeque, HashMap};

use cpu::Cpu;
use std::fs::File;
use std::io::{Read, BufReader};


fn main() {
    //let inst = Instruction::new(JUMP, Operand::NNN(0x200));

    //let mut opcodes: HashMap<&str, fn(Operand)> = HashMap::new();
    //opcodes.insert(CLEAR_SCREEN, clean_screen);
    //opcodes.insert(JUMP, jump);
    //opcodes["clear screen"](Operand::None);
    //opcodes[inst.name](inst.operand);

    println!("Hello, world!: {}", ENTRY_POINT);
    let mut cpu = Cpu::new();

    let mut f = File::open("roms/ibm-logo.ch8").unwrap();
    cpu.load(f);
    cpu.run();
    // read rom and test

    //cpu.run();
}


// opcodes

