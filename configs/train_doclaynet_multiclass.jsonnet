local train_base = import 'train_base.libsonnet';

local fixed_size_resize = 1e6;

train_base + {
  model_out_dir: "checkpoints/doclaynet_binary_r50_unet",

  color_labels: {
    type: "json",
    label_json_file: "data/doclaynet_multiclass/color_labels.json",
  },

  train_dataset: {
    type: "image_csv",
    csv_filename: "data/doclaynet_multiclass/train.csv",
    base_dir: "data/doclaynet_multiclass",
    repeat_dataset: 1,
    compose: {
      transforms: [{type: "fixed_size_resize", output_size: fixed_size_resize}],
    }
  },

  val_dataset: {
    type: "image_csv",
    csv_filename: "data/doclaynet_multiclass/val.csv",
    base_dir: "data/doclaynet_multiclass",
    compose: {
      transforms: [{type: "fixed_size_resize", output_size: fixed_size_resize}]
    }
  },

  num_epochs: 100,
  optimizer: {
        lr: 1e-4
  },
  batch_size: 6,
  evaluate_every_epoch: 1,

  early_stopping: {
    patience: 10,
  },
}

