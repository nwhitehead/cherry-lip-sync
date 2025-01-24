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
        let x = self.proj.forward(x);
        x
    }
}
