use symphonium::{SymphoniumLoader, DecodedAudio, ResampleQuality};
use std::vec::Vec;

const AUDIO_SAMPLERATE: u32 = 16000;
const WINDOW_TIME: f32 = 25e-3;
const HOP_TIME: f32 = 10e-3;
const WINDOW_LENGTH: usize = ((AUDIO_SAMPLERATE as f32) * WINDOW_TIME) as usize;
const HOP_LENGTH: usize = ((AUDIO_SAMPLERATE as f32) * HOP_TIME) as usize;

pub struct Pipeline {
    buffer: Vec<f32>,
    sample: DecodedAudio,
    position: usize,
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
        Self { buffer: b, sample, position: 0 }
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
                }
            }
            self.position += HOP_LENGTH;
            v
        }
    }
}