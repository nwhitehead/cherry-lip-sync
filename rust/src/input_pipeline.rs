use symphonium::{SymphoniumLoader, DecodedAudio, ResampleQuality};
use std::vec::Vec;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn::prelude::TensorData;
use rustfft::{FftPlanner, Fft, num_complex::Complex};
use std::sync::Arc;

use crate::hann::hann_window;

const AUDIO_SAMPLERATE: u32 = 16000;
const WINDOW_TIME: f32 = 25e-3;
const HOP_TIME: f32 = 10e-3;
const WINDOW_LENGTH: usize = ((AUDIO_SAMPLERATE as f32) * WINDOW_TIME) as usize;
const HOP_LENGTH: usize = ((AUDIO_SAMPLERATE as f32) * HOP_TIME) as usize;

pub struct Pipeline {
    buffer: Vec<f32>,
    sample: DecodedAudio,
    position: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl Pipeline {
    pub fn new(filename: &String) -> Self {
        let mut loader = SymphoniumLoader::new();
        let sample = loader
            .load(&filename, Some(AUDIO_SAMPLERATE), ResampleQuality::High, None)
            .expect("Should be able to load audio into memory");
        let mut b = Vec::with_capacity(sample.frames());
        b.resize(sample.frames(), 0.0);
        let num = sample.fill_channel(0, 0, &mut b);
        assert_eq!(num, Ok(sample.frames()));
        // Use FftPlanner to time implementations and record best for our size
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(WINDOW_LENGTH);
        Self { buffer: b, sample, position: 0, fft }
    }

    /// Get current window position (seconds)
    pub fn position(&self) -> f32 {
        (self.position as f32) / (AUDIO_SAMPLERATE as f32)
    }

    /// Determine if we are done reading samples
    pub fn done(&self) -> bool {
        self.position >= self.buffer.len()
    }

    /// Get next window of samples
    pub fn next(&mut self) -> Vec<f32> {
        if let Some(slice) = self.buffer.get(self.position..self.position + WINDOW_LENGTH) {
            self.position += HOP_LENGTH;
            Vec::from(slice)
        } else {
            let mut v = Vec::with_capacity(WINDOW_LENGTH);
            for i in 0..WINDOW_LENGTH {
                if let Some(elem) = self.buffer.get(self.position + i) {
                    v.push(*elem);
                } else {
                    v.push(0.0);
                }
            }
            self.position += HOP_LENGTH;
            v
        }
    }

    /// Get next window of processed samples
    pub fn processed<B: Backend>(&mut self) -> Tensor<B, 1> {
        // Setup Burn backend
        let device = Default::default();
        let samples = self.next();
        let x = Tensor::<B, 1>::from_data(TensorData::new(samples, [WINDOW_LENGTH]), &device);
        let hann = Tensor::<B, 1>::from_data(TensorData::new(hann_window(WINDOW_LENGTH), [WINDOW_LENGTH]), &device);
        let hann_x = x * hann;
        let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; WINDOW_LENGTH];
        // Fill real part of buffer with hann_x data
        for p in hann_x.clone().to_data().iter().enumerate() {
            buffer[p.0] = Complex{ re: p.1, im: 0.0f32 };
        }
        self.fft.process(&mut buffer);
        // Buffer now contains actual FFT results
        //dbg!(&buffer);
        hann_x
    }
}
