use burn_import::onnx::ModelGen;
use burn_import::burn::graph::RecordType;

fn main() {
    ModelGen::new()
        .input("./model/model_simple.onnx")
        .out_dir("./model/")
        .record_type(RecordType::Bincode)
        .half_precision(false)
        .embed_states(true)
        .run_from_script();
}
