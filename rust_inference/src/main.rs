use crate::model::ModelConfig;
use crate::model::Model;
use burn::backend::NdArray;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::NamedMpkFileRecorder;
use burn::module::Module;

mod model;

fn main() {
    type Backend = burn::backend::NdArray<>;

    let device = Default::default();
    let args = LoadArgs::new("./model-2-80-dropout.pt".into()).with_debug_print();
    let recorder  = PyTorchFileRecorder::<FullPrecisionSettings>::default();

    //let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(args.clone(), &device).expect("Should decode state successfully");
    let model = ModelConfig::new().init::<Backend>(&device).load_record(record);
    // model
    //     .clone()
    //     .save_file("test", &recorder)
    //     .expect("Save the model");
    // model
    //     .load_file("model-2-80-dropout.pt", &recorder, &device)
    //     .expect("Load the model");
    println!("Debug of model");
    println!("{}", model);
    println!("{:?}", args);
}
