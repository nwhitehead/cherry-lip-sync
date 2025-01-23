
use clap::Parser;
use symphonium::{SymphoniumLoader, ResampleQuality};
use input_pipeline::Pipeline;

mod input_pipeline;

static MODEL_BYTES: &[u8] = include_bytes!("../model/model.bin");

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
    let sample = Pipeline::new(&args.input);
}
