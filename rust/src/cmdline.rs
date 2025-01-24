
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use clap::Parser;
use crate::input_pipeline::{Pipeline, HOP_TIME};
use crate::model::ModelConfig;

mod input_pipeline;
mod hann;
mod model;

static MODEL_BYTES: &[u8] = include_bytes!("../model/model.bin");

type Backend = burn::backend::NdArray;

/// Analyze audio input and generate lip sync timing information output
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to input audio
    #[arg(short, long)]
    input: String,

    /// Path to output file to generate
    #[arg(short, long)]
    output: String,

    /// Desired FPS of output frames
    #[arg(short, long, default_value_t = 30.0)]
    fps: f32,

    /// Filter single frame output frames
    #[arg(long, default_value_t = false)]
    filter: bool,
}

fn main() {
    println!("LipSync");
    let args = Args::parse();
    dbg!(&args);
    // Load model
    let device = Default::default();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(MODEL_BYTES.to_vec(), &device)
        .expect("Should decode state successfully");
    let config = ModelConfig::new();
    let model = config
        .init::<Backend>(&device)
        .load_record(record);
    // Open input audio
    let mut sample = Pipeline::<Backend>::new(&args.input);
    // Process with model
    let mels = sample.batch_mel();
    println!("{}", mels.clone().slice([100..101, 0..26]));
    println!("{:?}", mels.clone().shape());
    let y = model.forward(mels.clone().unsqueeze());
    let predicted = y.clone().argmax(2).flatten::<1>(0, 2);
    println!("{:?}", y.clone().shape());
    println!("{:?}", predicted.clone().shape());
    println!("{}", y.clone().slice([0..1, 100..101, 0..12]));
    println!("{}", predicted);
    let visemes = predicted.into_data().into_vec::<i64>().expect("Able to convert tensor to vector");
    let frames = visemes.len() - config.lookahead;
    let sampled_frames = ((frames as f32) * HOP_TIME * args.fps).round() as usize;
    let mut last_viseme = -1;
    let mut dur = 2;
    for frame in 0..sampled_frames {
        let t = (frame as f32) / args.fps;
        let frame_original = (t / HOP_TIME).round() as usize + config.lookahead;
        let viseme = visemes[frame_original];
        if dur > 1 {
            if viseme == last_viseme {
                dur += 1;
            } else {
                dur = 1;
                last_viseme = viseme;
            }
        } else {
            dur += 1;
        }
        let actual_viseme = if args.filter {
            last_viseme
        } else {
            viseme
        };
        println!("t={:.4} frame={} frame0={} --\t {}", t, frame, frame_original, actual_viseme);
    }
}
