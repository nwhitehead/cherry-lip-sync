use crate::model::ModelConfig;
use burn::module::Module;
use burn::record::{NamedMpkFileRecorder, BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

mod model;

fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();

    // We need to remap the GRU weights because PyTorch saves them as concatenated matrices and
    // we have manually split them up into r/z/n.
    let args = LoadArgs::new("../data/model-2-80-split.ptx".into())
        .with_key_remap("net\\.1\\.weight_ih_l0.r", "gru1.reset_gate.input_transform.weight")
        .with_key_remap("net\\.1\\.weight_ih_l0.z", "gru1.update_gate.input_transform.weight")
        .with_key_remap("net\\.1\\.weight_ih_l0.n", "gru1.new_gate.input_transform.weight")
        .with_key_remap("net\\.1\\.weight_hh_l0.r", "gru1.reset_gate.hidden_transform.weight")
        .with_key_remap("net\\.1\\.weight_hh_l0.z", "gru1.update_gate.hidden_transform.weight")
        .with_key_remap("net\\.1\\.weight_hh_l0.n", "gru1.new_gate.hidden_transform.weight")
        .with_key_remap("net\\.1\\.bias_ih_l0.r", "gru1.reset_gate.input_transform.bias")
        .with_key_remap("net\\.1\\.bias_ih_l0.z", "gru1.update_gate.input_transform.bias")
        .with_key_remap("net\\.1\\.bias_ih_l0.n", "gru1.new_gate.input_transform.bias")
        .with_key_remap("net\\.1\\.bias_hh_l0.r", "gru1.reset_gate.hidden_transform.bias")
        .with_key_remap("net\\.1\\.bias_hh_l0.z", "gru1.update_gate.hidden_transform.bias")
        .with_key_remap("net\\.1\\.bias_hh_l0.n", "gru1.new_gate.hidden_transform.bias")
        .with_key_remap("net\\.4\\.weight_ih_l0.r", "gru2.reset_gate.input_transform.weight")
        .with_key_remap("net\\.4\\.weight_ih_l0.z", "gru2.update_gate.input_transform.weight")
        .with_key_remap("net\\.4\\.weight_ih_l0.n", "gru2.new_gate.input_transform.weight")
        .with_key_remap("net\\.4\\.weight_hh_l0.r", "gru2.reset_gate.hidden_transform.weight")
        .with_key_remap("net\\.4\\.weight_hh_l0.z", "gru2.update_gate.hidden_transform.weight")
        .with_key_remap("net\\.4\\.weight_hh_l0.n", "gru2.new_gate.hidden_transform.weight")
        .with_key_remap("net\\.4\\.bias_ih_l0.r", "gru2.reset_gate.input_transform.bias")
        .with_key_remap("net\\.4\\.bias_ih_l0.z", "gru2.update_gate.input_transform.bias")
        .with_key_remap("net\\.4\\.bias_ih_l0.n", "gru2.new_gate.input_transform.bias")
        .with_key_remap("net\\.4\\.bias_hh_l0.r", "gru2.reset_gate.hidden_transform.bias")
        .with_key_remap("net\\.4\\.bias_hh_l0.z", "gru2.update_gate.hidden_transform.bias")
        .with_key_remap("net\\.4\\.bias_hh_l0.n", "gru2.new_gate.hidden_transform.bias")
        .with_key_remap("net\\.6\\.(.*)", "proj.$1")
        .with_debug_print();

    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(args.clone(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new().init::<Backend>(&device).load_record(record);

    // Save model in MessagePack format with full precision
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file("../data/model", &recorder)
        .expect("Should be able to save the model");
    // Save in bin format
    let recorder = BinFileRecorder::<HalfPrecisionSettings>::new();
    model
        .clone()
        .save_file("../data/model-half", &recorder)
        .expect("Should be able to save the model");
    // Now save in half precision
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    model
        .clone()
        .save_file("../data/model-half", &recorder)
        .expect("Should be able to save the model");
    println!("Model converted from PyTorch to Rust binary format");
}
