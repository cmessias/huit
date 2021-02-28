use super::constants::instruction::*;
use super::constants::hardware::{MEMORY_SIZE, ENTRY_POINT, SCREEN_HEIGHT, SCREEN_WIDTH, STACK_SIZE};

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pixel {
    Black,
    White,
}

fn flip(pixel: &Pixel) -> Pixel {
    match pixel {
        Pixel::White => Pixel::Black,
        Pixel::Black => Pixel::White,
    }
}

struct Opcode(u16);

struct Address(u16);


fn to_nnn(opcode: Opcode) -> u16 {
    return opcode.0 & 0x0FFF;
}

fn to_nn(opcode: Opcode) -> u8 {
    return (opcode.0 & 0x00FF) as u8;
}

#[derive(Clone, Copy, Debug)]
pub enum Operand {
    None,
    X(usize),
    XY(usize, usize),
    XNN(usize, u8),
    NNN(u16),
    XYN(usize, usize, usize),
}

struct Instruction {
    function: fn(&mut Cpu, Operand),
    operand: Operand,
}

impl Instruction {
    pub fn new(function: fn(&mut Cpu, Operand), operand: Operand) -> Instruction {
        Instruction { function, operand }
    }
}

pub struct Cpu {
    pub memory: [u8; MEMORY_SIZE],
    display: [Pixel; SCREEN_WIDTH * SCREEN_HEIGHT],
    pc: usize,
    idx: usize,
    stack: VecDeque<usize>,
    registers: [u8; 16],
    _delay_timer: u8,
    _sound_timer: u8,
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu {
            memory: [0; MEMORY_SIZE],
            display: [Pixel::Black; SCREEN_WIDTH * SCREEN_HEIGHT],
            pc: ENTRY_POINT,
            idx: 0,
            stack: VecDeque::with_capacity(STACK_SIZE),
            registers: [0; 16],
            _delay_timer: 0,
            _sound_timer: 0,
        }
    }

    pub fn load(&mut self, mut rom: File) {
        let buffer = &mut self.memory[ENTRY_POINT..];
        rom.read(buffer);
    }
    pub fn run(&mut self) {
        for i in 0..5000 {
            let opcode = self.fetch();
            let instruction = self.decode(opcode);
            self.execute(instruction);


        }

        for y in 0..SCREEN_HEIGHT {
            for x in 0..SCREEN_WIDTH {
                if self.get_pixel(x, y) == Pixel::White {
                    print!(" ");
                } else {
                    print!("x");
                }
            }
            print!("\n");
        }
    }

    fn fetch(&mut self) -> Opcode {
        let byte1 = self.memory[self.pc] as u16;
        let byte2 = self.memory[self.pc + 1] as u16;
        let opcode = byte1 << 8 | byte2;

        self.pc += 2;
        return Opcode(opcode);
    }


    fn decode(&mut self, opcode: Opcode) -> Instruction {
        let a = ((opcode.0 & 0xF000) >> 12) as u8;
        let b = ((opcode.0 & 0x0F00) >> 8) as u8;
        let c = ((opcode.0 & 0x00F0) >> 4) as u8;
        let d = (opcode.0 & 0x000F) as u8;

        match (a, b, c, d) {
            (0, 0, 0xE, 0) => Instruction { function: Cpu::clear, operand: Operand::None },
            (1, _, _, _) => Instruction { function: Cpu::jmp, operand: Operand::NNN(to_nnn(opcode)) },
            (6, x, _, _) => Instruction { function: Cpu::setr, operand: Operand::XNN(x as usize, to_nn(opcode)) },
            (7, x, _, _) => Instruction { function: Cpu::add, operand: Operand::XNN(x as usize, to_nn(opcode)) },
            (0xA, _, _, _) => Instruction { function: Cpu::seti, operand: Operand::NNN(to_nnn(opcode)) },
            (0xD, x, y, n) => Instruction { function: Cpu::draw, operand: Operand::XYN(x as usize, y as usize, n as usize) },
            _ => panic!("unknown instruction: {}", opcode.0)
        }
    }

    fn execute(&mut self, inst: Instruction) {
        (inst.function)(self, inst.operand);
    }

    fn clear(&mut self, _: Operand) {
        println!("screen clear");
        for i in 0..(SCREEN_HEIGHT * SCREEN_WIDTH) {
            self.display[i] = Pixel::White;
        }
    }

    fn jmp(&mut self, op: Operand) {
        if let Operand::NNN(nnn) = op {
            self.pc = nnn as usize;
        }
    }

    fn call(&mut self, op: Operand) {
        if let Operand::NNN(nnn) = op {
            println!("calling function at {}", nnn);
            self.stack.push_front(self.pc);
            self.pc = nnn as usize;
        }
    }

    fn ret(&mut self, _op: Operand) {
        if let Some(ret_addr) = self.stack.pop_front() {
            self.pc = ret_addr;
        }
    }

    fn skipx_eq(&mut self, op: Operand) {
        if let Operand::XNN(x, nn) = op {
            if self.registers[x] == nn {
                self.pc += 2;
            }
        }
    }

    fn skipx_neq(&mut self, op: Operand) {
        if let Operand::XNN(x, nn) = op {
            if self.registers[x] != nn {
                self.pc += 2;
            }
        }
    }

    fn skipxy_eq(&mut self, op: Operand) {
        if let Operand::XY(x, y) = op {
            if self.registers[x] == self.registers[y] {
                self.pc += 2;
            }
        }
    }

    fn skipxy_neq(&mut self, op: Operand) {
        if let Operand::XY(x, y) = op {
            if self.registers[x] != self.registers[y] {
                self.pc += 2;
            }
        }
    }

    fn setr(&mut self, op: Operand) {
        if let Operand::XNN(x, nn) = op {
            self.registers[x] = nn;
        }
    }

    fn add(&mut self, op: Operand) {
        if let Operand::XNN(x, nn) = op {
            self.registers[x] += nn;
        }
    }

    fn seti(&mut self, op: Operand) {
        if let Operand::NNN(nnn) = op {
            self.idx = nnn as usize;
        }
    }

    fn draw(&mut self, op: Operand) {
        if let Operand::XYN(x, y, n) = op {
            println!("x: {}, y: {}, SW: {}, SH: {}", x, y, SCREEN_WIDTH as u8, SCREEN_HEIGHT as u8);
            let x = self.wrapping_x_on_screen(self.registers[x]);
            let y = self.wrapping_y_on_screen(self.registers[y]);
            println!("x: {}, y: {}, SW: {}, SH: {}", x, y, SCREEN_WIDTH as u8, SCREEN_HEIGHT as u8);
            self.registers[0xF] = 0;

            /*
            (0..n).for_each(|_| {
                let row = self.memory[self.idx + n];
                for i in 0..8 {
                    let sprite = (row >> (7 - i)) & 0x1;
                    if sprite == 1 {
                        if self.get_pixel(x, y) == Pixel::White {
                            self.registers[0xF] = 1;
                        }
                        self.pixel_flip(x, y);
                    }
                    x += 1;
                }
                y += 1;
            })
             */

            (0..n).for_each(|n: usize| {
                if y + n >= SCREEN_HEIGHT {
                    return;
                }
                let row = self.memory[self.idx + n];
                (0..8).for_each(|i| {
                    if x + i >= SCREEN_WIDTH {
                        return;
                    }
                    let sprite = (row >> (7 - i)) & 0x1;
                    if sprite == 1 {
                        if self.get_pixel(x + i, y + n) == Pixel::White {
                            self.registers[0xF] = 1;
                        }
                        self.pixel_flip(x + i, y + n);
                    }
                });
            });
        }
    }

    fn wrapping_x_on_screen(&self, x: u8) -> usize {
        (x % (SCREEN_WIDTH as u8)) as usize
    }

    fn wrapping_y_on_screen(&self, y: u8) -> usize {
        (y % (SCREEN_HEIGHT as u8)) as usize
    }

    fn get_pixel(&self, x: usize, y: usize) -> Pixel {
        self.display[x + y * SCREEN_WIDTH]
    }

    fn set_pixel(&mut self, x: usize, y: usize, pixel: Pixel) {
        self.display[x + y * SCREEN_WIDTH] = pixel;
    }

    fn pixel_flip(&mut self, x: usize, y: usize) {
        let pixel = x + y * SCREEN_WIDTH;
        self.display[pixel] = flip(&self.display[pixel])
    }
}


trait Bla {
    fn jump(&self, inst: Operand);
}


#[cfg(test)]
mod tests {
    use super::Cpu;
    use crate::constants::hardware::{ENTRY_POINT, SCREEN_WIDTH, SCREEN_HEIGHT};
    use crate::cpu::{Instruction, Operand, Pixel, to_nnn};

    #[test]
    fn should_clear_screen() {
        let mut cpu = Cpu::new();
        cpu.display = [Pixel::Black; SCREEN_WIDTH * SCREEN_HEIGHT];
        let inst = Instruction { function: Cpu::clear, operand: Operand::None };
        cpu.execute(inst);

        assert_eq!(cpu.display, [Pixel::White; SCREEN_WIDTH * SCREEN_HEIGHT])
    }

    #[test]
    fn should_jump_to_nnn() {
        let mut cpu = Cpu::new();
        let source_addr = ENTRY_POINT + 4;
        let inst = Instruction { function: Cpu::jmp, operand: Operand::NNN(source_addr as u16) };
        cpu.execute(inst);

        assert_eq!(cpu.pc, source_addr);
    }

    #[test]
    fn should_set_vx_to_nn() {
        let mut cpu = Cpu::new();
        let inst = Instruction { function: Cpu::setr, operand: Operand::XNN(0, 10) };
        cpu.execute(inst);

        assert_eq!(cpu.registers[0], 10);
    }

    #[test]
    fn should_add_nn_to_vx() {
        let mut cpu = Cpu::new();
        cpu.registers[0] = 10;
        let inst = Instruction { function: Cpu::add, operand: Operand::XNN(0, 10) };
        cpu.execute(inst);

        assert_eq!(cpu.registers[0], 20);
    }

    #[test]
    fn should_set_idx_to_nnn() {
        let mut cpu = Cpu::new();
        let source_addr = ENTRY_POINT + 4;
        let inst = Instruction { function: Cpu::seti, operand: Operand::NNN(source_addr as u16) };
        cpu.execute(inst);

        assert_eq!(cpu.idx, source_addr);
    }

    #[test]
    fn should_draw_sprite() {
        let mut cpu = Cpu::new();

        cpu.idx = 0x300;
        cpu.memory[cpu.idx] = 0x3C;
        cpu.memory[cpu.idx + 1] = 0xC3;
        cpu.memory[cpu.idx + 2] = 0xFF;
        cpu.registers[0x1] = 1;
        cpu.registers[0x2] = 2;

        let inst = Instruction { function: Cpu::draw, operand: Operand::XYN(0x1, 0x2, 3) };
        cpu.execute(inst);

        print!("begin display\n");
        for y in 0..SCREEN_HEIGHT {
            for x in 0..SCREEN_WIDTH {
                if cpu.get_pixel(x, y) == Pixel::White {
                    print!("*");
                } else {
                    print!(" ");
                }
            }
            print!("\n");
        }
        assert!(!cpu.display.iter().all(|&p| p == Pixel::Black));
    }
}