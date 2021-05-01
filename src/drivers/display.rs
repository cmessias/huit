use crate::constants::hardware::{
    INTERNAL_SCREEN_HEIGHT, INTERNAL_SCREEN_WIDTH, SCALE_FACTOR, SCREEN_SIZE,
};
use crate::cpu::Pixel;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::video::Window;
use sdl2::Sdl;

pub struct DisplayDriver {
    canvas: Canvas<Window>,
}

impl DisplayDriver {
    pub fn new(sdl_context: &Sdl, screen_width: u32, screen_height: u32) -> DisplayDriver {
        let video_subsystem = sdl_context.video().unwrap();

        let window = video_subsystem
            .window("Huit", screen_width, screen_height)
            .position_centered()
            .build()
            .unwrap();

        let mut canvas = window.into_canvas().present_vsync().build().unwrap();

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas.present();

        return DisplayDriver { canvas };
    }

    pub fn draw(&mut self, display: [Pixel; SCREEN_SIZE]) {
        for y in 0..INTERNAL_SCREEN_HEIGHT {
            for x in 0..INTERNAL_SCREEN_WIDTH {
                let color = self.get_color(&display, x, y);
                self.canvas.set_draw_color(color);
                let x = (x * (SCALE_FACTOR as usize)) as i32;
                let y = (y * SCALE_FACTOR) as i32;

                self.canvas
                    .fill_rect(Rect::new(x, y, SCALE_FACTOR as u32, SCALE_FACTOR as u32))
                    .expect("Could not draw on screen");
            }
        }
        self.canvas.present();
    }

    fn get_color(&self, display: &[Pixel; SCREEN_SIZE], x: usize, y: usize) -> Color {
        return if DisplayDriver::get_pixel(&display, x, y) == Pixel::White {
            Color::RGB(250, 250, 250)
        } else {
            Color::RGB(0, 0, 0)
        };
    }

    fn get_pixel(display: &[Pixel; SCREEN_SIZE], x: usize, y: usize) -> Pixel {
        return display[x + y * INTERNAL_SCREEN_WIDTH];
    }
}
