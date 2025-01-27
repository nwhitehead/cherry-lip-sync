use crate::model::ModelConfig;
use burn::module::Module;
use burn::module::Param;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn::record::{
    BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder, Recorder,
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

mod model;

#[derive(Module, Debug)]
struct FloatTensor<B: Backend, const D: usize> {
    test: Param<Tensor<B, D>>,
}

fn load_tensor<B: Backend, const D: usize>(path: &str) -> Tensor<B, D> {
    let trecord: FloatTensorRecord<B, D> = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(path.into(), &Default::default())
        .expect("Load tensor");
    return trecord.test.val();
}

fn save_tensor<B: Backend, const D: usize>(x: Tensor<B, D>, path: &str) {
    BinFileRecorder::<FullPrecisionSettings>::new()
        .record(x, path.into())
        .expect("Should decode state successfully");
}

fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();

    // We need to remap the GRU weights because PyTorch saves them as concatenated matrices and
    // we have manually split them up into r/z/n.
    let args = LoadArgs::new("./model/model.ptx".into())
        .with_key_remap(
            "(net\\.1\\.)(.+)",
            "bnorm.$2",
        )
        .with_key_remap(
            "net\\.4\\.weight_ih_l0.r",
            "gru1.reset_gate.input_transform.weight",
        )
        .with_key_remap(
            "net\\.4\\.weight_ih_l0.z",
            "gru1.update_gate.input_transform.weight",
        )
        .with_key_remap(
            "net\\.4\\.weight_ih_l0.n",
            "gru1.new_gate.input_transform.weight",
        )
        .with_key_remap(
            "net\\.4\\.weight_hh_l0.r",
            "gru1.reset_gate.hidden_transform.weight",
        )
        .with_key_remap(
            "net\\.4\\.weight_hh_l0.z",
            "gru1.update_gate.hidden_transform.weight",
        )
        .with_key_remap(
            "net\\.4\\.weight_hh_l0.n",
            "gru1.new_gate.hidden_transform.weight",
        )
        .with_key_remap(
            "net\\.4\\.bias_ih_l0.r",
            "gru1.reset_gate.input_transform.bias",
        )
        .with_key_remap(
            "net\\.4\\.bias_ih_l0.z",
            "gru1.update_gate.input_transform.bias",
        )
        .with_key_remap(
            "net\\.4\\.bias_ih_l0.n",
            "gru1.new_gate.input_transform.bias",
        )
        .with_key_remap(
            "net\\.4\\.bias_hh_l0.r",
            "gru1.reset_gate.hidden_transform.bias",
        )
        .with_key_remap(
            "net\\.4\\.bias_hh_l0.z",
            "gru1.update_gate.hidden_transform.bias",
        )
        .with_key_remap(
            "net\\.4\\.bias_hh_l0.n",
            "gru1.new_gate.hidden_transform.bias",
        )
        .with_key_remap(
            "net\\.7\\.weight_ih_l0.r",
            "gru2.reset_gate.input_transform.weight",
        )
        .with_key_remap(
            "net\\.7\\.weight_ih_l0.z",
            "gru2.update_gate.input_transform.weight",
        )
        .with_key_remap(
            "net\\.7\\.weight_ih_l0.n",
            "gru2.new_gate.input_transform.weight",
        )
        .with_key_remap(
            "net\\.7\\.weight_hh_l0.r",
            "gru2.reset_gate.hidden_transform.weight",
        )
        .with_key_remap(
            "net\\.7\\.weight_hh_l0.z",
            "gru2.update_gate.hidden_transform.weight",
        )
        .with_key_remap(
            "net\\.7\\.weight_hh_l0.n",
            "gru2.new_gate.hidden_transform.weight",
        )
        .with_key_remap(
            "net\\.7\\.bias_ih_l0.r",
            "gru2.reset_gate.input_transform.bias",
        )
        .with_key_remap(
            "net\\.7\\.bias_ih_l0.z",
            "gru2.update_gate.input_transform.bias",
        )
        .with_key_remap(
            "net\\.7\\.bias_ih_l0.n",
            "gru2.new_gate.input_transform.bias",
        )
        .with_key_remap(
            "net\\.7\\.bias_hh_l0.r",
            "gru2.reset_gate.hidden_transform.bias",
        )
        .with_key_remap(
            "net\\.7\\.bias_hh_l0.z",
            "gru2.update_gate.hidden_transform.bias",
        )
        .with_key_remap(
            "net\\.7\\.bias_hh_l0.n",
            "gru2.new_gate.hidden_transform.bias",
        )
        .with_key_remap("net\\.9\\.(.*)", "proj.$1")
        .with_debug_print();

    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(args.clone(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new()
        .init::<Backend>(&device)
        .load_record(record);

    // Save model in binary format with full precision
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file("../model/model", &recorder)
        .expect("Should be able to save the model");
    println!("Model converted from PyTorch to Rust binary format");

    // Now convert melbank for audio input processing
    let x = load_tensor::<Backend, 2>("../model/melbank.pt");
    save_tensor(x, "../model/melbank");
    
    // Now convert test input/output
    let x = load_tensor::<Backend, 3>("../model/test_in.pt");
    save_tensor(x, "../model/test_in");
    let out = load_tensor::<Backend, 3>("../model/test_out.pt");
    save_tensor(out, "../model/test_out");
    println!("Test tensors converted from PyTorch to Rust binary format");
}
