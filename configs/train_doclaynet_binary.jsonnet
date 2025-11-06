local train_base = import 'train_base.libsonnet';

/*
  Ровно как в примере: используем fixed_size_resize с выходной площадью.
  1e6 ≈ ~1000×1000 — надёжный старт для страниц.
*/
local fixed_size_resize = 1e6;

train_base + {
  model_out_dir: "checkpoints/doclaynet_binary_r50_unet",

  color_labels: {
    type: "json",
    label_json_file: "data/doclaynet_binary/color_labels.json",
  },

  train_dataset: {
    type: "image_csv",
    csv_filename: "data/doclaynet_binary/train.csv",
    base_dir: "data/doclaynet_binary",
    repeat_dataset: 1,
    compose: {
      transforms: [{type: "fixed_size_resize", output_size: fixed_size_resize}],
    }
  },

  val_dataset: {
    type: "image_csv",
    csv_filename: "data/doclaynet_binary/val.csv",
    base_dir: "data/doclaynet_binary",
    compose: {
      transforms: [{type: "fixed_size_resize", output_size: fixed_size_resize}]
    }
  },

  num_epochs: 100,
  optimizer: {
        lr: 1e-3
  },
  batch_size: 4,
  evaluate_every_epoch: 2,

  // При необходимости можно раскомментировать раннюю остановку
  // early_stopping: {
  //   patience: 10,
  // },
}

