use crate::model::Model;
use crate::model::ModelConfig;
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::NamedMpkFileRecorder;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

mod model;

fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();
    let args = LoadArgs::new("./model-2-80-dropout.ptx".into())
        .with_key_remap("net\\.1\\.weight_ih_l0.r", "gru1.reset_gate.input_transform")
        .with_key_remap("net\\.1\\.weight_ih_l0.z", "gru1.update_gate.input_transform")
        .with_key_remap("net\\.1\\.weight_ih_l0.n", "gru1.new_gate.input_transform")
        .with_key_remap("net\\.1\\.weight_hh_l0.r", "gru1.reset_gate.hidden_transform")
        .with_key_remap("net\\.1\\.weight_hh_l0.z", "gru1.update_gate.hidden_transform")
        .with_key_remap("net\\.1\\.weight_hh_l0.n", "gru1.new_gate.hidden_transform")
        .with_key_remap("net\\.1\\.weight_ih_l1.r", "gru2.reset_gate.input_transform")
        .with_key_remap("net\\.1\\.weight_ih_l1.z", "gru2.update_gate.input_transform")
        .with_key_remap("net\\.1\\.weight_ih_l1.n", "gru2.new_gate.input_transform")
        .with_key_remap("net\\.1\\.weight_hh_l1.r", "gru2.reset_gate.hidden_transform")
        .with_key_remap("net\\.1\\.weight_hh_l1.z", "gru2.update_gate.hidden_transform")
        .with_key_remap("net\\.1\\.weight_hh_l1.n", "gru2.new_gate.hidden_transform")
        .with_key_remap("net\\.3\\.(.*)", "proj.$1")
        .with_debug_print();
    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();

    //let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(args.clone(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new()
        .init::<Backend>(&device)
        .load_record(record);
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
