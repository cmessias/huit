use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::log::Category::Input;
use sdl2::Sdl;

pub struct InputDriver {
    event_pump: sdl2::EventPump,
}

pub enum PollResult {
    Keys(Vec<usize>),
    Quit,
}

impl InputDriver {
    pub fn new(sdl_context: &Sdl) -> InputDriver {
        let event_pump = sdl_context.event_pump().unwrap();
        return InputDriver { event_pump };
    }

    pub fn poll(&mut self) -> PollResult {
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return PollResult::Quit,
                _ => continue,
            }
        }

        let pressed_keys: Vec<usize> = self
            .event_pump
            .keyboard_state()
            .pressed_scancodes()
            .filter_map(Keycode::from_scancode)
            .filter_map(InputDriver::get_chip8_key)
            .collect();

        return PollResult::Keys(pressed_keys);
    }

    fn get_chip8_key(key: Keycode) -> Option<usize> {
        use sdl2::keyboard::Keycode::*;
        match key {
            Num1 => Some(0x1),
            Num2 => Some(0x2),
            Num3 => Some(0x3),
            Num4 => Some(0xC),
            Q => Some(0x4),
            W => Some(0x5),
            E => Some(0x6),
            R => Some(0xD),
            A => Some(0x7),
            S => Some(0x8),
            D => Some(0x9),
            F => Some(0xE),
            Z => Some(0xA),
            X => Some(0x0),
            C => Some(0xB),
            V => Some(0xF),
            _ => None,
        }
    }
}
