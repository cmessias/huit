use super::constants::instruction::*;
use super::constants::hardware::{MEMORY_SIZE, ENTRY_POINT, SCREEN_HEIGHT, SCREEN_WIDTH, STACK_SIZE};

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use rand::Rng;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pixel {
    Black,
    White,
}

impl Pixel {
    fn flip(&self) -> Pixel {
        match &self {
            Pixel::White => Pixel::Black,
            Pixel::Black => Pixel::White,
        }
    }
}

type Instruction = Box<dyn FnOnce(&mut Cpu)>;

fn to_nnn(opcode: u16) -> usize {
    return (opcode & 0x0FFF) as usize;
}

fn to_nn(opcode: u16) -> u8 {
    return (opcode & 0x00FF) as u8;
}

fn to_x(x: u8) -> usize {
    return x as usize;
}

fn to_y(y: u8) -> usize {
    return y as usize;
}

pub struct Cpu {
    pub memory: [u8; MEMORY_SIZE],
    display: [Pixel; SCREEN_WIDTH * SCREEN_HEIGHT],
    pc: usize,
    idx: usize,
    stack: VecDeque<usize>,
    v: [u8; 16],
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
            v: [0; 16],
            _delay_timer: 0,
            _sound_timer: 0,
        }
    }

    pub fn load(&mut self, mut rom: File) {
        let buffer = &mut self.memory[ENTRY_POINT..];
        rom.read(buffer);
    }
    pub fn run(&mut self) {
        for i in 0..20 {
            self.run_cycle();
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

    fn run_cycle(&mut self) {
        let opcode = self.fetch();
        let instruction = self.decode(opcode);
        self.execute(instruction);
    }

    fn fetch(&mut self) -> u16 {
        let byte1 = self.memory[self.pc] as u16;
        let byte2 = self.memory[self.pc + 1] as u16;
        let opcode = byte1 << 8 | byte2;

        self.pc += 2;
        return opcode;
    }

    fn decode(&self, opcode: u16) -> Instruction {
        let a = ((opcode & 0xF000) >> 12) as u8;
        let b = ((opcode & 0x0F00) >> 8) as u8;
        let c = ((opcode & 0x00F0) >> 4) as u8;
        let d = (opcode & 0x000F) as u8;

        return match (a, b, c, d) {
            (0, 0, 0xE, 0) => Box::new(|cpu| clr(cpu)),
            (0, 0, 0xE, 0xE) => Box::new(|cpu| ret(cpu)),
            (1, _, _, _) => Box::new(move |cpu| jmp(cpu, to_nnn(opcode))),
            (2, _, _, _) => Box::new(move |cpu| call(cpu, to_nnn(opcode))),
            (3, x, _, _) => Box::new(move |cpu| skipx_eq(cpu, to_x(x), to_nn(opcode))),
            (4, x, _, _) => Box::new(move |cpu| skipx_neq(cpu, to_x(x), to_nn(opcode))),
            (5, x, y, 0) => Box::new(move |cpu| skipxy_eq(cpu, to_x(x), to_y(y))),
            (6, x, _, _) => Box::new(move |cpu| setx(cpu, to_x(x), to_nn(opcode))),
            (7, x, _, _) => Box::new(move |cpu| addx(cpu, to_x(x), to_nn(opcode))),
            (8, x, y, 0) => Box::new(move |cpu| setxy(cpu, to_x(x), to_y(y))),
            (8, x, y, 1) => Box::new(move |cpu| orxy(cpu, to_x(x), to_y(y))),
            (8, x, y, 2) => Box::new(move |cpu| andxy(cpu, to_x(x), to_y(y))),
            (8, x, y, 3) => Box::new(move |cpu| xorxy(cpu, to_x(x), to_y(y))),
            (8, x, y, 4) => Box::new(move |cpu| addxyc(cpu, to_x(x), to_y(y))),
            (8, x, y, 5) => Box::new(move |cpu| subxy(cpu, to_x(x), to_y(y))),
            (8, x, y, 6) => Box::new(move |cpu| shiftxyr(cpu, to_x(x), to_y(y))),
            (8, x, y, 7) => Box::new(move |cpu| subyx(cpu, to_x(x), to_y(y))),
            (8, x, y, 0xE) => Box::new(move |cpu| shiftxyl(cpu, to_x(x), to_y(y))),
            (9, x, y, 0) => Box::new(move |cpu| skipxy_neq(cpu, to_x(x), to_y(y))),
            (0xA, _, _, _) => Box::new(move |cpu| seti(cpu, to_nnn(opcode))),
            (0xB, _, _, _) => Box::new(move |cpu| jmp0(cpu, to_nnn(opcode))),
            (0xD, x, y, n) => Box::new(move |cpu| draw(cpu, x as usize, y as usize, n as usize)),
            (0xF, x, 1, 0xE) => Box::new(move |cpu| addi(cpu, to_x(x))),
            (0xF, x, 3, 3) => Box::new(move |cpu| bcd(cpu, to_x(x))),
            (0xF, x, 5, 5) => Box::new(move |cpu| stxi(cpu, to_x(x))),
            (0xF, x, 6, 5) => Box::new(move |cpu| ldxi(cpu, to_x(x))),
            _ => panic!("unknown instruction: {}", opcode)
        };
    }

    fn execute(&mut self, instruction: Instruction) {
        instruction(self);
    }


    fn wrapx_on_screen(&self, x: u8) -> usize {
        (x % (SCREEN_WIDTH as u8)) as usize
    }

    fn wrapy_on_screen(&self, y: u8) -> usize {
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
        self.display[pixel] = self.display[pixel].flip()
    }
}

fn clr(cpu: &mut Cpu) {
    for i in 0..(SCREEN_HEIGHT * SCREEN_WIDTH) {
        cpu.display[i] = Pixel::White;
    }
}

fn jmp(cpu: &mut Cpu, nnn: usize) {
    cpu.pc = nnn;
}

fn setx(cpu: &mut Cpu, x: usize, nn: u8) {
    cpu.v[x] = nn;
}

fn addx(cpu: &mut Cpu, x: usize, nn: u8) {
    cpu.v[x] += nn;
}

fn seti(cpu: &mut Cpu, nnn: usize) {
    cpu.idx = nnn as usize;
}

fn draw(cpu: &mut Cpu, x: usize, y: usize, n: usize) {
    let x = cpu.wrapx_on_screen(cpu.v[x]);
    let y = cpu.wrapy_on_screen(cpu.v[y]);
    cpu.v[0xF] = 0;

    (0..n).for_each(|n: usize| {
        if y + n >= SCREEN_HEIGHT {
            return;
        }
        let row = cpu.memory[cpu.idx + n];
        (0..8).for_each(|i| {
            if x + i >= SCREEN_WIDTH {
                return;
            }
            let sprite = (row >> (7 - i)) & 1;
            if sprite == 1 {
                if cpu.get_pixel(x + i, y + n) == Pixel::White {
                    cpu.v[0xF] = 1;
                }
                cpu.pixel_flip(x + i, y + n);
            }
        });
    });
}

fn call(cpu: &mut Cpu, nnn: usize) {
    cpu.stack.push_front(cpu.pc);
    cpu.pc = nnn;
}

fn ret(cpu: &mut Cpu) {
    if let Some(ret_addr) = cpu.stack.pop_front() {
        cpu.pc = ret_addr;
    }
}

fn skipx_eq(cpu: &mut Cpu, x: usize, nn: u8) {
    if cpu.v[x] == nn {
        cpu.pc += 2;
    }
}

fn skipx_neq(cpu: &mut Cpu, x: usize, nn: u8) {
    if cpu.v[x] != nn {
        cpu.pc += 2;
    }
}

fn skipxy_eq(cpu: &mut Cpu, x: usize, y: usize) {
    if cpu.v[x] == cpu.v[y] {
        cpu.pc += 2;
    }
}

fn skipxy_neq(cpu: &mut Cpu, x: usize, y: usize) {
    if cpu.v[x] != cpu.v[y] {
        cpu.pc += 2;
    }
}

fn setxy(cpu: &mut Cpu, x: usize, y: usize) {
    cpu.v[x] = cpu.v[y];
}

fn orxy(cpu: &mut Cpu, x: usize, y: usize) {
    cpu.v[x] |= cpu.v[y];
}

fn andxy(cpu: &mut Cpu, x: usize, y: usize) {
    cpu.v[x] &= cpu.v[y];
}

fn xorxy(cpu: &mut Cpu, x: usize, y: usize) {
    cpu.v[x] ^= cpu.v[y];
}

fn addxyc(cpu: &mut Cpu, x: usize, y: usize) {
    let (result, carry) = cpu.v[x].overflowing_add(cpu.v[y]);
    cpu.v[x] = result;
    cpu.v[0xF] = carry as u8;
}

fn subxy(cpu: &mut Cpu, x: usize, y: usize) {
    let (result, borrow) = cpu.v[x].overflowing_sub(cpu.v[y]);
    cpu.v[x] = result;
    cpu.v[0xF] = borrow as u8;
}

fn subyx(cpu: &mut Cpu, x: usize, y: usize) {
    let (result, borrow) = cpu.v[y].overflowing_sub(cpu.v[x]);
    cpu.v[y] = result;
    cpu.v[0xF] = borrow as u8;
}

fn shiftxyr(cpu: &mut Cpu, x: usize, y: usize) {
    cpu.v[0xF] = cpu.v[y] & 1;
    cpu.v[x] = cpu.v[y] >> 1;
}

fn shiftxyl(cpu: &mut Cpu, x: usize, y: usize) {
    cpu.v[0xF] = (cpu.v[y] >> 7) & 1;
    cpu.v[x] = cpu.v[y] << 1;
}

#[allow(dead_code)]
fn shiftxr(cpu: &mut Cpu, x: usize) {
    cpu.v[0xF] = cpu.v[x] & 1;
    cpu.v[x] >>= 1;
}

#[allow(dead_code)]
fn shiftxl(cpu: &mut Cpu, x: usize) {
    cpu.v[0xF] = (cpu.v[x] >> 7) & 1;
    cpu.v[x] <<= 1;
}

fn jmp0(cpu: &mut Cpu, nnn: usize) {
    cpu.pc = (cpu.v[0] as usize) + nnn;
}

#[allow(dead_code)]
fn jmpx(cpu: &mut Cpu, x: usize, nnn: usize) {
    cpu.pc = (cpu.v[x] as usize) + nnn;
}

fn rand(cpu: &mut Cpu, x: usize, nn: u8) {
    cpu.v[x] = rand::thread_rng().gen_range(0..=nn) & nn;
}

fn addi(cpu: &mut Cpu, x: usize) {
    cpu.idx = cpu.idx.wrapping_add(cpu.v[x] as usize);
}

#[allow(dead_code)]
fn addic(cpu: &mut Cpu, x: usize) {
    cpu.idx = cpu.idx.wrapping_add(cpu.v[x] as usize);
    if cpu.idx > 0xFFF {
        cpu.v[0xF] = 1;
    }
}

fn bcd(cpu: &mut Cpu, x: usize) {
    cpu.memory[cpu.idx] = cpu.v[x] / 100;
    cpu.memory[cpu.idx + 1] = (cpu.v[x] / 10) % 10;
    cpu.memory[cpu.idx + 2] = cpu.v[x] % 10;
}

fn stxi(cpu: &mut Cpu, x: usize) {
    let memory = &mut cpu.memory[cpu.idx..=(cpu.idx + x)];
    let registers = &cpu.v[0..=x];
    memory.copy_from_slice(registers);
    cpu.idx += x + 1;
}

#[allow(dead_code)]
fn stx(cpu: &mut Cpu, x: usize) {
    let memory = &mut cpu.memory[cpu.idx..=(cpu.idx + x)];
    let registers = &cpu.v[0..=x];
    memory.copy_from_slice(registers);
}

fn ldxi(cpu: &mut Cpu, x: usize) {
    let registers = &mut cpu.v[0..=x];
    let memory = &cpu.memory[cpu.idx..=(cpu.idx + x)];
    registers.copy_from_slice(memory);
    cpu.idx += x + 1;
}

#[allow(dead_code)]
fn ldx(cpu: &mut Cpu, x: usize) {
    let registers = &mut cpu.v[0..=x];
    let memory = &cpu.memory[cpu.idx..=(cpu.idx + x)];
    registers.copy_from_slice(memory);
}

#[cfg(test)]
mod tests {
    use crate::constants::hardware::{ENTRY_POINT, SCREEN_WIDTH, SCREEN_HEIGHT};
    use crate::cpu::{Pixel, Cpu};
    use std::io::SeekFrom::End;

    #[test]
    fn should_clear_screen() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x00;
        cpu.memory[ENTRY_POINT + 1] = 0xE0;
        cpu.display = [Pixel::Black; SCREEN_WIDTH * SCREEN_HEIGHT];
        cpu.run_cycle();

        assert_eq!(cpu.display, [Pixel::White; SCREEN_WIDTH * SCREEN_HEIGHT])
    }

    #[test]
    fn should_jump_to_nnn() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x12;
        cpu.memory[ENTRY_POINT + 1] = 0x22;
        cpu.run_cycle();

        assert_eq!(cpu.pc, 0x222);
    }

    #[test]
    fn should_call_subroutine() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x2A;
        cpu.memory[ENTRY_POINT + 1] = 0xAA;
        cpu.run_cycle();

        assert_eq!(*cpu.stack.front().unwrap(), ENTRY_POINT + 2);
        assert_eq!(cpu.pc, 0xAAA);
    }

    #[test]
    fn should_return_from_subroutine() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x00;
        cpu.memory[ENTRY_POINT + 1] = 0xEE;
        cpu.stack.push_front(ENTRY_POINT);
        cpu.run_cycle();

        assert_eq!(cpu.pc, ENTRY_POINT);
    }

    #[test]
    fn should_set_vx_to_nn() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x61;
        cpu.memory[ENTRY_POINT + 1] = 0xFF;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0xFF);
    }

    #[test]
    fn should_add_nn_to_vx() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x71;
        cpu.memory[ENTRY_POINT + 1] = 0x98;
        cpu.v[1] = 1;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0x99);
    }

    #[test]
    fn should_set_idx_to_nnn() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0xA3;
        cpu.memory[ENTRY_POINT + 1] = 0x33;
        cpu.run_cycle();

        assert_eq!(cpu.idx, 0x333);
    }

    #[test]
    fn should_skipx_eq() {
        let mut cpu = Cpu::new();
        let nn = 0xAA;
        cpu.memory[ENTRY_POINT] = 0x31;
        cpu.memory[ENTRY_POINT + 1] = nn;
        cpu.v[1] = nn;
        cpu.run_cycle();

        assert_eq!(cpu.pc, ENTRY_POINT + 4);
    }


    #[test]
    fn should_skipx_neq() {
        let mut cpu = Cpu::new();
        let nn = 0xAA;
        cpu.memory[ENTRY_POINT] = 0x41;
        cpu.memory[ENTRY_POINT + 1] = nn;
        cpu.v[1] = nn + 1;
        cpu.run_cycle();

        assert_eq!(cpu.pc, ENTRY_POINT + 4);
    }

    #[test]
    fn should_skipxy_eq() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x51;
        cpu.memory[ENTRY_POINT + 1] = 0x20;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0xAA;
        cpu.run_cycle();

        assert_eq!(cpu.pc, ENTRY_POINT + 4);
    }

    #[test]
    fn should_skipxy_neq() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x91;
        cpu.memory[ENTRY_POINT + 1] = 0x20;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0xBB;
        cpu.run_cycle();

        assert_eq!(cpu.pc, ENTRY_POINT + 4);
    }

    #[test]
    fn should_setxy() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x20;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0xBB;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0xBB);
        assert_eq!(cpu.v[2], 0xBB);
    }

    #[test]
    fn should_orxy() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x21;
        cpu.v[1] = 0b10010011;
        cpu.v[2] = 0b11110001;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0b11110011);
        assert_eq!(cpu.v[2], 0b11110001)
    }

    #[test]
    fn should_andxy() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x22;
        cpu.v[1] = 0b10010011;
        cpu.v[2] = 0b11110001;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0b10010001);
        assert_eq!(cpu.v[2], 0b11110001)
    }

    #[test]
    fn should_xorxy() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x23;
        cpu.v[1] = 0b10010011;
        cpu.v[2] = 0b11110001;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0b01100010);
        assert_eq!(cpu.v[2], 0b11110001)
    }

    #[test]
    fn should_addxy() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x24;
        cpu.v[1] = 1;
        cpu.v[2] = 2;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 3);
        assert_eq!(cpu.v[2], 2);
        assert_eq!(cpu.v[0xF], 0);
    }

    #[test]
    fn should_addxy_carry() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x24;
        cpu.v[1] = 0xFF;
        cpu.v[2] = 2;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 1);
        assert_eq!(cpu.v[2], 2);
        assert_eq!(cpu.v[0xF], 1);
    }

    #[test]
    fn should_subxy() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x25;
        cpu.v[1] = 0xFF;
        cpu.v[2] = 1;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0xFE);
        assert_eq!(cpu.v[2], 1);
        assert_eq!(cpu.v[0xF], 0);
    }

    #[test]
    fn should_subxy_borrow() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x25;
        cpu.v[1] = 1;
        cpu.v[2] = 0xFF;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 2);
        assert_eq!(cpu.v[2], 0xFF);
        assert_eq!(cpu.v[0xF], 1);
    }

    #[test]
    fn should_subyx() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x27;
        cpu.v[1] = 1;
        cpu.v[2] = 0xFF;
        cpu.run_cycle();

        assert_eq!(cpu.v[2], 0xFE);
        assert_eq!(cpu.v[1], 1);
        assert_eq!(cpu.v[0xF], 0);
    }

    #[test]
    fn should_subyx_borrow() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x27;
        cpu.v[1] = 0xFF;
        cpu.v[2] = 1;
        cpu.run_cycle();

        assert_eq!(cpu.v[2], 2);
        assert_eq!(cpu.v[1], 0xFF);
        assert_eq!(cpu.v[0xF], 1);
    }

    #[test]
    fn should_shiftxyr() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x26;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0xFE;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0x7F);
        assert_eq!(cpu.v[2], 0xFE);
        assert_eq!(cpu.v[0xF], 0);
    }

    #[test]
    fn should_shiftxyr_carry() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x26;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0xFF;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0x7F);
        assert_eq!(cpu.v[2], 0xFF);
        assert_eq!(cpu.v[0xF], 1);
    }

    #[test]
    fn should_shiftxyl() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x2E;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0x7F;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0xFE);
        assert_eq!(cpu.v[2], 0x7F);
        assert_eq!(cpu.v[0xF], 0);
    }

    #[test]
    fn should_shiftxyl_carry() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0x81;
        cpu.memory[ENTRY_POINT + 1] = 0x2E;
        cpu.v[1] = 0xAA;
        cpu.v[2] = 0xFF;
        cpu.run_cycle();

        assert_eq!(cpu.v[1], 0xFE);
        assert_eq!(cpu.v[2], 0xFF);
        assert_eq!(cpu.v[0xF], 1);
    }

    #[test]
    fn should_jmp0() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0xB2;
        cpu.memory[ENTRY_POINT + 1] = 0x22;
        cpu.v[0] = 1;
        cpu.run_cycle();

        assert_eq!(cpu.pc, 0x223);
    }

    #[test]
    fn should_addi() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0xF1;
        cpu.memory[ENTRY_POINT + 1] = 0x1E;
        cpu.idx = 0xFFE;
        cpu.v[1] = 1;
        cpu.run_cycle();

        assert_eq!(cpu.idx, 0xFFF);
    }

    #[test]
    fn should_convert_bcd() {
        let mut cpu = Cpu::new();
        cpu.memory[ENTRY_POINT] = 0xF1;
        cpu.memory[ENTRY_POINT + 1] = 0x33;
        cpu.idx = ENTRY_POINT + 2;
        cpu.v[1] = 156;
        cpu.run_cycle();

        assert_eq!(cpu.memory[cpu.idx], 1);
        assert_eq!(cpu.memory[cpu.idx + 1], 5);
        assert_eq!(cpu.memory[cpu.idx + 2], 6);
    }

    #[test]
    fn should_stxi_all_registers() {
        let mut cpu = Cpu::new();
        let x: usize = 0xF;
        cpu.memory[ENTRY_POINT] = 0xF0 + x as u8;
        cpu.memory[ENTRY_POINT + 1] = 0x55;
        cpu.idx = ENTRY_POINT + 2;
        let initial_idx = cpu.idx;
        cpu.v = [1; 16];

        cpu.run_cycle();
        assert_eq!(cpu.idx, initial_idx + x + 1);
        cpu.idx = initial_idx;

        let memory_region = &cpu.memory[cpu.idx..=(cpu.idx + x)];
        assert_eq!(memory_region, &cpu.v);
    }

    #[test]
    fn should_stxi_up_to_x_registers() {
        let mut cpu = Cpu::new();
        let x = 9;
        cpu.memory[ENTRY_POINT] = 0xF0 + x as u8;
        cpu.memory[ENTRY_POINT + 1] = 0x55;
        cpu.idx = ENTRY_POINT + 2;
        let initial_idx = cpu.idx;
        cpu.v = [1; 16];

        cpu.run_cycle();
        assert_eq!(cpu.idx, initial_idx + x + 1);
        cpu.idx = initial_idx;

        let initial_memory_region = &cpu.memory[cpu.idx..=(cpu.idx + x)];
        let remaining_memory_region = &cpu.memory[(cpu.idx + x + 1)..=(cpu.idx + 0xF)];

        assert_eq!(initial_memory_region, &cpu.v[0..=x]);
        assert_eq!(remaining_memory_region, &[0u8; 6]);
    }

    #[test]
    fn should_ldxi_all_registers() {
        let mut cpu = Cpu::new();
        let x = 0xF;
        cpu.memory[ENTRY_POINT] = 0xF0 + x as u8;
        cpu.memory[ENTRY_POINT + 1] = 0x65;
        cpu.idx = ENTRY_POINT + 2;
        let initial_idx = cpu.idx;
        let memory_region = &mut cpu.memory[cpu.idx..=(cpu.idx + 0xF)];
        memory_region.copy_from_slice(&[1u8; 16]);

        cpu.run_cycle();
        assert_eq!(cpu.idx, initial_idx + x + 1);
        assert_eq!(cpu.v, [1u8; 16]);
    }

    #[test]
    fn should_ldxi_up_to_x_registers() {
        let mut cpu = Cpu::new();
        let x = 9;
        cpu.memory[ENTRY_POINT] = 0xF0 + x as u8;
        cpu.memory[ENTRY_POINT + 1] = 0x65;
        cpu.idx = ENTRY_POINT + 2;
        let initial_idx = cpu.idx;
        let memory_region = &mut cpu.memory[cpu.idx..=(cpu.idx + x)];
        memory_region.copy_from_slice(&[1u8; 10]);

        cpu.run_cycle();
        assert_eq!(cpu.idx, initial_idx + x + 1);

        let expected: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        assert_eq!(cpu.v, expected);
    }
}