use crate::model::Model;
use crate::model::ModelConfig;
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::NamedMpkFileRecorder;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::PrettyJsonFileRecorder;
use burn::prelude::Tensor;
use burn::prelude::Backend;
use burn::module::ConstantRecord;
use burn::module::Param;
use burn::module::ParamId;
use burn::record::Record;

mod model;

#[derive(Module, Debug)]
struct FloatTensor<B: Backend, const D: usize> {
    test: Param<Tensor<B, D>>,
}

fn load_tensor<B: Backend, const D: usize>(path: &str) -> Tensor<B, D> {
    let trecord: FloatTensorRecord<B, D> =
        PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load(path.into(), &Default::default())
            .expect("Load tensor");
    return trecord.test.val();
}

fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();

    let args = LoadArgs::new("./model-random3.ptx".into())
        .with_key_remap("net\\.0\\.weight_ih_l0.r", "gru1.reset_gate.input_transform.weight")
        .with_key_remap("net\\.0\\.weight_ih_l0.z", "gru1.update_gate.input_transform.weight")
        .with_key_remap("net\\.0\\.weight_ih_l0.n", "gru1.new_gate.input_transform.weight")
        .with_key_remap("net\\.0\\.weight_hh_l0.r", "gru1.reset_gate.hidden_transform.weight")
        .with_key_remap("net\\.0\\.weight_hh_l0.z", "gru1.update_gate.hidden_transform.weight")
        .with_key_remap("net\\.0\\.weight_hh_l0.n", "gru1.new_gate.hidden_transform.weight")
        .with_key_remap("net\\.0\\.bias_ih_l0.r", "gru1.reset_gate.input_transform.bias")
        .with_key_remap("net\\.0\\.bias_ih_l0.z", "gru1.update_gate.input_transform.bias")
        .with_key_remap("net\\.0\\.bias_ih_l0.n", "gru1.new_gate.input_transform.bias")
        .with_key_remap("net\\.0\\.bias_hh_l0.r", "gru1.reset_gate.hidden_transform.bias")
        .with_key_remap("net\\.0\\.bias_hh_l0.z", "gru1.update_gate.hidden_transform.bias")
        .with_key_remap("net\\.0\\.bias_hh_l0.n", "gru1.new_gate.hidden_transform.bias")
        .with_key_remap("net\\.3\\.(.*)", "proj.$1")
        .with_debug_print();

    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(args.clone(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new().init::<Backend>(&device).load_record(record);

    // let trecord: FloatTensorRecord<Backend, 2> =
    //     PyTorchFileRecorder::<FullPrecisionSettings>::new()
    //         .load("blah".into(), &device)
    //         .expect("Load tensor");
    // let x = trecord.into_item::<FullPrecisionSettings>();

    let x = load_tensor::<Backend, 3>("../data/test_in_0.pt");
    let y = model.forward(x.clone());
    let out = load_tensor::<Backend, 3>("../data/test_out_0.pt");
    // model
    //     .clone()
    //     .save_file("test", &recorder)
    //     .expect("Save the model");
    // model
    //     .load_file("model-2-80-dropout.pt", &recorder, &device)
    //     .expect("Load the model");
    println!("Debug of model");
    // model
    //     .clone()
    //     .save_file("test", &recorder)
    //     .expect("Save the model");
    println!("{}", model);
    // println!("{:?}", x.test.val());
    println!("input tensor = {:?}", x);
    println!("rust model output = {:?}", y);
    println!("torch model output = {:?}", out);
}
