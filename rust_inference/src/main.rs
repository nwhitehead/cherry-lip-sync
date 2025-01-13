use crate::model::ModelConfig;
use burn::backend::NdArray;

mod model;

fn main() {
    type Backend = burn::backend::NdArray<>;

    let device = Default::default();
    let model = ModelConfig::new().init::<Backend>(&device);
    println!("Debug of model");
    println!("{}", model);
}
