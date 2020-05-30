use ndarray::{Array1, Array2, Axis, Slice};
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::time::SystemTime;

use learn_rs::Tree;

#[derive(Debug, Deserialize)]
struct BostonRecord {
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64,
    b: f64,
    lstat: f64,
    medv: f64,
}

struct BostonDataSet {
    features: Array2<f64>,
    labels: Array1<f64>,
}

impl BostonDataSet {
    pub fn from(records: Vec<BostonRecord>) -> Self {
        let mut features = Vec::with_capacity(records.len());
        let mut labels = Vec::with_capacity(records.len());

        for r in records {
            let feats = &[
                r.crim, r.zn, r.indus, r.chas, r.nox, r.rm, r.age, r.dis, r.rad, r.tax, r.ptratio,
                r.b, r.lstat,
            ];
            features.push(feats.clone());
            labels.push(r.medv);
        }

        Self {
            features: Array2::from(features),
            labels: Array1::from(labels),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/boston.csv";
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut bfr = Vec::new();

    for result in rdr.deserialize() {
        let record: BostonRecord = result?;
        bfr.push(record);
    }

    let dataset = BostonDataSet::from(bfr);
    println!("Dataset size: {}", &dataset.features.nrows());

    let train_dataset = BostonDataSet {
        features: dataset
            .features
            .slice_axis(Axis(0), Slice::from(0..450))
            .to_owned(),
        labels: dataset
            .labels
            .slice_axis(Axis(0), Slice::from(0..450))
            .to_owned(),
    };

    let test_dataset = BostonDataSet {
        features: dataset
            .features
            .slice_axis(Axis(0), Slice::from(450..))
            .to_owned(),
        labels: dataset
            .labels
            .slice_axis(Axis(0), Slice::from(450..))
            .to_owned(),
    };

    let start_training = SystemTime::now();
    let decision_tree = Tree::new(
        train_dataset.features.clone(),
        train_dataset.labels.clone(),
        5,
    );
    let end_training = start_training.elapsed().unwrap().as_millis();
    println!("Training took {} ms", end_training);

    let start_prediction = SystemTime::now();
    let yhat_train = train_dataset
        .features
        .map_axis(Axis(1), |x| decision_tree.predict(x.to_owned()));
    let yhat_test = test_dataset
        .features
        .map_axis(Axis(1), |x| decision_tree.predict(x.to_owned()));
    let prediction_duration = start_prediction.elapsed().unwrap().as_millis();
    println!("Prediction took {} ms", prediction_duration);

    println!("yhat train shape: {:?}", yhat_train.shape());
    println!("yhat test shape: {:?}", yhat_test.shape());
    let mae_train = (&yhat_train - &train_dataset.labels)
        .map(|y| y.abs())
        .mean()
        .unwrap();
    let mae_test = (&yhat_test - &test_dataset.labels)
        .map(|y| y.abs())
        .mean()
        .unwrap();

    println!("Tree size: {}", &decision_tree.size());
    println!("MAE Train: {:?}", mae_train);
    println!("MAE Test: {:?}", mae_test);
    Ok(())
}
