use crate::model::ModelConfig;
use burn::module::Module;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use std::vec::Vec;

mod model;

static MODEL_BYTES: &[u8] = include_bytes!("../model/model.bin");
static TENSOR_IN_BYTES: &[u8] = include_bytes!("../model/test_in.bin");
static TENSOR_OUT_BYTES: &[u8] = include_bytes!("../model/test_out.bin");

fn load_tensor<B: Backend, const D: usize>(data: Vec<u8>) -> Tensor<B, D> {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    return recorder
        .load(data, &Default::default())
        .expect("Load tensor");
}

fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(MODEL_BYTES.to_vec(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new()
        .init::<Backend>(&device)
        .load_record(record);
    println!("Model loaded");
    let x = load_tensor::<Backend, 3>(TENSOR_IN_BYTES.to_vec());
    let y = model.forward(x.clone());
    let out = load_tensor::<Backend, 3>(TENSOR_OUT_BYTES.to_vec());
    println!("{}", model);
    println!("input tensor = {:?}", x);
    println!("rust model output = {:?}", y);
    println!("torch model output = {:?}", out);
    out.to_data().assert_approx_eq(&y.to_data(), 3);
    println!("Passed numeric test");
}
