use burn::{
    nn::{
        gru::{Gru, GruConfig},
        Linear, LinearConfig,
    },
    // nn::{
    //     Dropout, DropoutConfig,
    // }
    prelude::{Module, Tensor, Config, Backend},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub gru1: Gru<B>,
    pub gru2: Gru<B>,
    pub proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "26")]
    input_size: usize,
    #[config(default = "80")]
    hidden_size: usize,
    #[config(default = "12")]
    num_classes: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            gru1: GruConfig::new(self.input_size, self.hidden_size, /*bias=*/true).init(device),
            gru2: GruConfig::new(self.hidden_size, self.hidden_size, /*bias=*/true).init(device),
            proj: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, time, input_size]
    ///   - Output [batch_size, time, output_classes]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = input;
        let x = self.gru1.forward(x, None);
        let x = self.gru2.forward(x, None);
        let x = self.proj.forward(x);
        x
    }
}