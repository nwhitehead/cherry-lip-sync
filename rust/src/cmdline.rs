
use clap::Parser;
use input_pipeline::Pipeline;
use burn::prelude::Tensor;
use burn::prelude::TensorData;

type Backend = burn::backend::NdArray;

mod input_pipeline;
mod hann;

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
    // Setup Burn backend
    let device = Default::default();
    // Open input audio
    let mut sample = Pipeline::new(&args.input);
    while !sample.done() {
        let swin = sample.next();
        println!("t={}", sample.position());
        let sz = swin.len();
        let x = Tensor::<Backend, 1>::from_data(TensorData::new(swin, [sz]), &device);
    }
}
