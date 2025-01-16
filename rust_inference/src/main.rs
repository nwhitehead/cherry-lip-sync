use crate::model::ModelConfig;
use burn::module::Module;
use burn::record::NamedMpkFileRecorder;
use burn::record::{FullPrecisionSettings, Recorder};

mod model;


fn main() {
    type Backend = burn::backend::NdArray;

    let device = Default::default();

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load("../data/model".into(), &device)
        .expect("Should decode state successfully");
    let model = ModelConfig::new().init::<Backend>(&device).load_record(record);

    println!("{}", model);
}
