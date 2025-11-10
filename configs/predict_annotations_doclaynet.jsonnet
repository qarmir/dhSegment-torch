local train = import 'train_doclaynet_multiclass.jsonnet';

{
  data: {
    type: "folder",
    folder: "data/doclaynet_multiclass/to_predict",
    pre_processing: train.val_dataset.compose
  },

  model: {
    type: "training_config",
    model: train.model,
    color_labels: train.color_labels,
    dataset: train.train_dataset,
    model_state_dict: "checkpoints/doclaynet_binary/train_model_checkpoint_iter=52501.pth"
  },

  batch_size: train.batch_size,

  writer: {
    type: "image",
    overwrite: true
  }
}