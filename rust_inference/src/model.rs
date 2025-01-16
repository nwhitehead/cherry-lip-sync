use burn::{
    nn::{
        gru::{Gru, GruConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::{Module, Tensor, Config, Backend},
    record::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub gru1: Gru<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "2")]
    input_size: usize,
    #[config(default = "3")]
    hidden_size: usize,
    #[config(default = "1")]
    num_classes: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            gru1: GruConfig::new(self.input_size, self.hidden_size, /*bias=*/true).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, time, input_size]
    ///   - Output [batch_size, time, output_classes]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, sequence_length, input_size] = input.dims();
        let x = input;
        let x = self.gru1.forward(x, None);
        x
    }
}