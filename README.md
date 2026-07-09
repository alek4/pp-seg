# pp-seg

**P**otted **P**lants **Seg**mentation - binary plant segmentation in a greenhouse setting.

## Dataset

Public dataset from Roboflow, collected in a greenhouse and annotated in COCO
segmentation format (`greenhouse-3-1/`). After removing unannotated images: 695 train / 208 valid / 106 test. All images resized to 512x512. Foreground covers only ~2.5% of pixels, which drives most of the design choices (loss, metrics).

## Approach

1. **Baseline** - `smp.Unet` (ResNet34, ImageNet pretrained). Strong but 24.4M params.
2. **BinarySegCNN** - flat encoder-decoder, 185K params. Struggles with thin
   structures (stems, partial leaves); changing the loss doesn't fix it, so the
   bottleneck is the architecture.
3. **BinaryUNet** - custom UNet with skip connections, configurable depth and width.
   Trained with BCE + Dice loss to handle the class imbalance.
4. **Grid search** - 3-fold CV over lr, batch size, base filters, depth, selecting
   on plant IoU. Winner: `base_filters=32, depth=2, lr=1e-3, batch_size=8`.

## Results

Test-set plant IoU, threshold tuned on the validation split:

| Model                          | Params | Plant IoU |
|--------------------------------|-------:|----------:|
| smp.Unet (ResNet34, ImageNet)  |  24.4M |    0.9102 |
| BinarySegCNN (BCEDice)         |   185K |    0.8418 |
| BinaryUNet f16 d3 (BCEDice)    |   488K |    0.9079 |
| **BinaryUNet f32 d2 (final)**  |   472K |    0.9092 |

## Structure

- `ppseg.ipynb` - data exploration, preprocessing,
  models, grid search, final training and evaluation, results summary
- `models/` - `BinarySegCNN`, `BinaryUNet`
- `utils/` - dataset and loaders (`data.py`), training loop (`train.py`), losses
  (`losses.py`), evaluation protocol (`evaluate.py`), plotting (`viz.py`),
  grid search (`kfold_gridsearch.py`)
- `outputs/` - checkpoints, normalization stats, grid search results (gitignored)

## Running it

Sections of the notebook are independent: each one creates its own dataloaders, so you can run any section from a fresh kernel without running the ones before it.
Training cells are the slow part; every trained model is evaluated from its checkpoint, so evaluation cells can be re-run without retraining.

## Future work

- Improve generalization with a bigger dataset covering different plant species
- Revisit the pretrained route, which should generalize better with more varied data
- Build a new dataset and fine-tune the model on pseudo-labels generated with SAM 3
