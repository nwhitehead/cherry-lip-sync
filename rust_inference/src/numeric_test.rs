use crate::model::ModelConfig;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use burn::prelude::Tensor;
use burn::prelude::Backend;
use burn::module::Param;

mod model;

static MODEL_BYTES: &[u8] = include_bytes!("../../data/model.bin");

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

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(MODEL_BYTES.to_vec(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new().init::<Backend>(&device).load_record(record);

    let x = load_tensor::<Backend, 3>("../data/test_in.pt");
    let y = model.forward(x.clone());
    let out = load_tensor::<Backend, 3>("../data/test_out.pt");
    println!("{}", model);
    println!("input tensor = {:?}", x);
    println!("rust model output = {:?}", y);
    println!("torch model output = {:?}", out);
    out.to_data().assert_approx_eq(&y.to_data(), 3);
    println!("Passed numeric test");
}
