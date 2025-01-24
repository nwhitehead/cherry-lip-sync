
use clap::Parser;
use crate::input_pipeline::Pipeline;
use crate::hann::hann_window;

mod input_pipeline;
mod hann;

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
}

fn main() {
    println!("LipSync");
    let args = Args::parse();
    dbg!(&args);
    // Open input audio
    let mut sample = Pipeline::<Backend>::new(&args.input);
    // Process with model
    let res = sample.batch_process();
    println!("{}", res.slice([50..51, 0..26]));
}
