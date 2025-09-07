# 🐱🐶 Cat vs Dog Classifier using CNN with SE Block

This project is a **binary image classifier** that distinguishes between **cats and dogs** using a **deep convolutional neural network** enhanced with **Squeeze-and-Excitation (SE) blocks**, `BatchNormalization`, `Dropout`, and `ImageDataGenerator` for data augmentation.

---

## 🚀 Features

- ✅ Based on VGG-style CNN architecture
- ✅ Integrated SE Block to improve channel-wise feature weighting
- ✅ Includes BatchNormalization and Dropout for regularization
- ✅ Uses Keras `ImageDataGenerator` for real-time data augmentation
- ✅ Clear train/validation/test split
- ✅ Accuracy, loss and AUC plots included

---

## 🗂 Dataset Structure

Make sure your dataset is structured like this:

```
dataset/
├── train/
│   ├── CAT/
│   └── DOG/
└── test/
    ├── CAT/
    └── DOG/
```

- **Training set**: 25,000 images
- **Validation set**: 5,000 images (automatically split from training using `validation_split`)
- **Test set**: 800 images for final evaluation

---

## 🧠 Model Architecture

- Input: 224×224 RGB images
- Several convolutional blocks with ReLU, BatchNormalization, and MaxPooling
- `SpatialDropout2D` in deeper layers
- One SE Block for enhancing important feature channels
- Global Average Pooling followed by Dense layers
- Output: 1 neuron with sigmoid activation for binary classification

---

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- matplotlib
- pillow
- (Optional) Jupyter Notebook for interactive experimentation

Install with:

```bash
pip install tensorflow matplotlib pillow
```

---

## 🧪 Training

```python
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1050,
)
```

The model automatically uses data augmentation and monitors validation accuracy/loss.

---

## 📊 Accuracy, Loss and AUC Plots

The script includes accuracy/loss/auc plotting after training:

```python
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
...
```

---

## 🧬 Future Improvements

- Add **CutMix** or **MixUp** augmentation
- Convert model to `TFLite` for mobile deployment
- Use **Grad-CAM** for visualizing model attention

---

## 📁 Output

- Best model saved as `best_model.keras`
- You can evaluate it on test data:

```python
model.evaluate(test_generator)
```

---

## 🤝 Contributing

Feel free to fork, modify, and submit PRs!

---

## 📜 License

This project is licensed under the MIT License.
