use symphonium::{SymphoniumLoader, DecodedAudio, ResampleQuality};
use std::vec::Vec;

const AUDIO_SAMPLERATE: u32 = 16000;
const WINDOW_TIME: f32 = 25e-3;
const HOP_TIME: f32 = 10e-3;

pub struct Pipeline {
    buffer: Vec<f32>,
    sample: DecodedAudio,
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
        Self { buffer: b, sample }
    }
}