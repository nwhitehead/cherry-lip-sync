
use std::vec::Vec;
use std::f32::consts::PI;

pub fn hann_window(length: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..length {
        let x = (i as f32) / (length as f32);
        let phase = x * PI;
        v.push(phase);
    }
    v
}
