use burn::{
    nn::{
        gru::{Gru, GruConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
    },
    prelude::{Backend, Config, Module, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub bnorm: BatchNorm<B, 1>,
    pub gru1: Gru<B>,
    pub gru2: Gru<B>,
    pub proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "26")]
    pub input_size: usize,
    #[config(default = "80")]
    pub hidden_size: usize,
    #[config(default = "12")]
    pub num_classes: usize,
    #[config(default = "0.2")]
    pub dropout: f64,
    #[config(default = "3")]
    pub lookahead: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            bnorm: BatchNormConfig::new(self.input_size).init(device),
            gru1: GruConfig::new(self.input_size, self.hidden_size, /*bias=*/ true).init(device),
            gru2: GruConfig::new(self.hidden_size, self.hidden_size, /*bias=*/ true).init(device),
            proj: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, time, input_size]
    ///   - Output [batch_size, time, output_classes]
    #[allow(dead_code)]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = input;
        // BatchNorm needs N C T but our standard order is N T C
        let x = self.bnorm.forward(x.permute([0, 2, 1])).permute([0, 2, 1]);
        let x = self.gru1.forward(x, None);
        let x = self.gru2.forward(x, None);
        self.proj.forward(x)
    }
}

#[cfg(test)]
mod tests {

    use crate::model::ModelConfig;
    use burn::module::Module;
    use burn::prelude::Backend;
    use burn::prelude::Tensor;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use std::vec::Vec;

    fn load_tensor<B: Backend, const D: usize>(data: Vec<u8>) -> Tensor<B, D> {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        return recorder
            .load(data, &Default::default())
            .expect("Load tensor");
    }

    #[test]
    fn test_output() {
        static MODEL_BYTES: &[u8] = include_bytes!("../model/model.bin");
        static TENSOR_IN_BYTES: &[u8] = include_bytes!("../model/test_in.bin");
        static TENSOR_OUT_BYTES: &[u8] = include_bytes!("../model/test_out.bin");

        type MyBackend = burn::backend::NdArray;
        let device = Default::default();

        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(MODEL_BYTES.to_vec(), &device)
            .expect("Should decode state successfully");
        let model = ModelConfig::new()
            .init::<MyBackend>(&device)
            .load_record(record);
        println!("Model loaded");
        let x = load_tensor::<MyBackend, 3>(TENSOR_IN_BYTES.to_vec());
        let y = model.forward(x.clone());
        let out = load_tensor::<MyBackend, 3>(TENSOR_OUT_BYTES.to_vec());
        println!("{}", model);
        println!("input tensor = {:?}", x);
        println!("rust model output = {:?}", y);
        println!("torch model output = {:?}", out);
        out.to_data().assert_approx_eq(&y.to_data(), 3);
    }
}
