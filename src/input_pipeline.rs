use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn::prelude::TensorData;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;
use std::vec::Vec;
use symphonium::{ResampleQuality, SymphoniumLoader};

use crate::hann::hann_window;

const AUDIO_SAMPLERATE: u32 = 16000;
const MELS: usize = 13;
const WINDOW_TIME: f32 = 25e-3;
pub const HOP_TIME: f32 = 10e-3;
const WINDOW_LENGTH: usize = ((AUDIO_SAMPLERATE as f32) * WINDOW_TIME) as usize;
const HOP_LENGTH: usize = ((AUDIO_SAMPLERATE as f32) * HOP_TIME) as usize;
const FFT_LENGTH: usize = WINDOW_LENGTH / 2 + 1;

static MELBANK_BYTES: &[u8] = include_bytes!("../model/melbank.bin");

fn load_tensor<B: Backend, const D: usize>(data: Vec<u8>) -> Tensor<B, D> {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    recorder
        .load(data, &Default::default())
        .expect("Load tensor")
}

pub struct Pipeline<B: Backend> {
    buffer: Vec<f32>,
    position: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    hann: Tensor<B, 1>,
    melbanks: Tensor<B, 2>,
    device: B::Device,
}

impl<B: Backend> Pipeline<B> {
    pub fn new(filename: &String) -> Self {
        let mut loader = SymphoniumLoader::new();
        let sample = loader
            .load(
                filename,
                Some(AUDIO_SAMPLERATE),
                ResampleQuality::High,
                None,
            )
            .expect("Should be able to load audio into memory");
        let mut b = Vec::with_capacity(sample.frames());
        b.resize(sample.frames(), 0.0);
        let num = sample.fill_channel(0, 0, &mut b);
        assert_eq!(num, Ok(sample.frames()));
        // Use FftPlanner to time implementations and record best for our size
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(WINDOW_LENGTH);
        // Setup tensors so we don't have to recompute them every frame
        let device = Default::default();
        let hann = Tensor::<B, 1>::from_data(
            TensorData::new(hann_window(WINDOW_LENGTH), [WINDOW_LENGTH]),
            &device,
        );
        let melbanks = load_tensor::<B, 2>(MELBANK_BYTES.to_vec());
        Self {
            buffer: b,
            position: 0,
            fft,
            hann,
            melbanks,
            device,
        }
    }

    /// Get next window of samples
    pub fn next(&mut self) -> Vec<f32> {
        if let Some(slice) = self
            .buffer
            .get(self.position..self.position + WINDOW_LENGTH)
        {
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
    pub fn next_power(&mut self) -> Tensor<B, 2> {
        let samples = self.next();
        let x = Tensor::<B, 1>::from_data(TensorData::new(samples, [WINDOW_LENGTH]), &self.device);
        let hann_x = x * self.hann.clone();
        let mut input_buffer = vec![0.0f32; WINDOW_LENGTH];
        let mut output_buffer = vec![
            Complex {
                re: 0.0f32,
                im: 0.0f32
            };
            FFT_LENGTH
        ];
        // Fill real part of buffer with hann_x data
        for p in hann_x.clone().to_data().iter().enumerate() {
            input_buffer[p.0] = p.1;
        }
        self.fft
            .process(&mut input_buffer, &mut output_buffer)
            .expect("Should be able to compute FFT");
        // Buffer now contains actual FFT results
        let power = output_buffer.iter().map(Complex::norm_sqr).collect();
        Tensor::<B, 2>::from_data(TensorData::new(power, [1, FFT_LENGTH]), &self.device)
    }

    /// Batch process
    pub fn batch_mel(&mut self) -> Tensor<B, 2> {
        // Compute output frames
        let sz = (self.buffer.len() - (WINDOW_LENGTH - 1)).div_ceil(HOP_LENGTH);
        let mut pwr = Tensor::<B, 2>::zeros([sz, FFT_LENGTH], &self.device);
        for i in 0..sz {
            let r = self.next_power();
            pwr = pwr.slice_assign([i..(i + 1), 0..FFT_LENGTH], r);
        }
        // Convert from FFT bins to Mel banks
        let pwr = pwr.matmul(self.melbanks.clone());
        // Now do log power conversion
        let pwr = pwr.clamp_min(1e-10).log() / std::f32::consts::LN_10 * 10.0;
        // Now do derivatives
        let der = pwr.clone().pad((0, 0, 2, 1), 0.0);
        let d_off0 = der.clone().slice([0..sz, 0..MELS]);
        let d_off1 = der.clone().slice([1..sz + 1, 0..MELS]);
        let d_off2 = pwr.clone();
        let d_off3 = der.clone().slice([3..sz + 3, 0..MELS]);
        // Concatenate powers and smeared derivatives
        Tensor::cat(
            vec![
                pwr,
                d_off2 * 0.5 + d_off3 * 0.5 - d_off0 * 0.5 - d_off1 * 0.5,
            ],
            1,
        )
    }
}
