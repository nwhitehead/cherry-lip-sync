use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        gru::{Gru, GruConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // conv1: Conv2d<B>,
    // conv2: Conv2d<B>,
    // pool: AdaptiveAvgPool2d,
    dropout1: Dropout,
    dropout2: Dropout,
    gru1: Gru<B>,
    gru2: Gru<B>,
    proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "26")]
    input_size: usize,
    #[config(default = "12")]
    num_classes: usize,
    #[config(default = "100")]
    hidden_size: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

/*
    nn.Dropout(p=0.2),
    nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=0.2),
    SelectItem(0),
    nn.Linear(hidden_size, num_classes),

*/

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            dropout1: DropoutConfig::new(self.dropout).init(),
            gru1: GruConfig::new(self.input_size, self.hidden_size, /*bias=*/true).init(device),
            dropout2: DropoutConfig::new(self.dropout).init(),
            gru2: GruConfig::new(self.input_size, self.hidden_size, /*bias=*/true).init(device),
            proj: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
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
        let x = self.dropout1.forward(x);
        let x = self.gru1.forward(x, None);
        let x = self.dropout2.forward(x);
        let x = self.gru2.forward(x, None);
        let x = self.proj.forward(x);
        x
    }
}