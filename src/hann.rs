use std::f32::consts::PI;
use std::vec::Vec;

pub fn hann_window(length: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..length {
        let x = (i as f32) / (length as f32);
        let phase = x * PI;
        let sphase = phase.sin();
        v.push(sphase * sphase);
    }
    v
}
