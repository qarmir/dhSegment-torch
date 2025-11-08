local encoders = import './models/encoders.libsonnet';
local decoders = import './models/decoders.libsonnet';
local fixed_size_resize = 1e6;

{
  model: {
      encoder: encoders.resnet50,
      decoder: decoders.pan {
          decoder_channels_size: 512
      }
  },

  optimizer: {
    type: "adamw",
    lr: 1e-3,
    weight_decay: 1e-4
  },

  metrics: [
    ["miou", "iou"],
    ["iou", {
          "type": "iou",
          "average": null
      }],
    "precision"
  ],

  val_metric: "+miou",

  lr_scheduler: {
    type: "exponential",
    gamma: 0.99998
  },

  train_checkpoint: {type: "iteration", checkpoints_to_keep: 2},
  val_checkpoint: {checkpoints_to_keep: 2},

  num_data_workers: 4,
  track_train_metrics: false,
  loggers: [{
    type: "tensorboard",
    log_every: 50,
    log_images_every: 100
  }],

  model_out_dir: "checkpoints/doclaynet_binary",

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
  batch_size: 6,
  evaluate_every_epoch: 1,

  early_stopping: {
    patience: 10,
  },
}

