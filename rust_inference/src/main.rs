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

fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();

    let args = LoadArgs::new("./model-2-80-dropout.ptx".into())
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
        .with_key_remap("net\\.1\\.weight_ih_l1.r", "gru2.reset_gate.input_transform.weight")
        .with_key_remap("net\\.1\\.weight_ih_l1.z", "gru2.update_gate.input_transform.weight")
        .with_key_remap("net\\.1\\.weight_ih_l1.n", "gru2.new_gate.input_transform.weight")
        .with_key_remap("net\\.1\\.weight_hh_l1.r", "gru2.reset_gate.hidden_transform.weight")
        .with_key_remap("net\\.1\\.weight_hh_l1.z", "gru2.update_gate.hidden_transform.weight")
        .with_key_remap("net\\.1\\.weight_hh_l1.n", "gru2.new_gate.hidden_transform.weight")
        .with_key_remap("net\\.1\\.bias_ih_l1.r", "gru2.reset_gate.input_transform.bias")
        .with_key_remap("net\\.1\\.bias_ih_l1.z", "gru2.update_gate.input_transform.bias")
        .with_key_remap("net\\.1\\.bias_ih_l1.n", "gru2.new_gate.input_transform.bias")
        .with_key_remap("net\\.1\\.bias_hh_l1.r", "gru2.reset_gate.hidden_transform.bias")
        .with_key_remap("net\\.1\\.bias_hh_l1.z", "gru2.update_gate.hidden_transform.bias")
        .with_key_remap("net\\.1\\.bias_hh_l1.n", "gru2.new_gate.hidden_transform.bias")
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

    let trecord: FloatTensorRecord<Backend, 2> =
        PyTorchFileRecorder::<FullPrecisionSettings>::new()
            .load("blah".into(), &device)
            .expect("Load tensor");
    let orig_tensor = Tensor::<Backend, 2>::zeros([0, 0], &device);
    let tensor = FloatTensor::<Backend, 2> {
        test: Param::initialized(ParamId::new(), orig_tensor),
    };
    let tensor = tensor.load_record(trecord);
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
    println!("{:?}", tensor.test.val());
}
