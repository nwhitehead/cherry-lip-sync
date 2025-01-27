
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

fn viseme_to_str(data: i64) -> &'static str {
    [ "D", "B", "I", "G", "H", "A", "X", "E", "K", "J", "C", "F"][data as usize]
}

fn main() {
    println!("LipSync");
    let args = Args::parse();
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
    let mut outfile_data = Vec::<String>::new();
    // Process with model
    let mels = sample.batch_mel();
    let y = model.forward(mels.clone().unsqueeze());
    let predicted = y.clone().argmax(2).flatten::<1>(0, 2);
    let visemes = predicted.into_data().into_vec::<i64>().expect("Able to convert tensor to vector");
    let frames = visemes.len() - config.lookahead;
    let sampled_frames = ((frames as f32) * HOP_TIME * args.fps).round() as usize;
    let mut last_viseme = -1;
    let mut dur = 2;
    let mut previous_output_viseme = -1;
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
        if actual_viseme != previous_output_viseme {
            let output_str = format!("{:.3}\t{}", t, viseme_to_str(actual_viseme));
            println!("{}", output_str.clone());
            outfile_data.push(output_str);
        }
        previous_output_viseme = actual_viseme;
    }
    let _ = std::fs::write(args.output, outfile_data.join("\n"));
}
