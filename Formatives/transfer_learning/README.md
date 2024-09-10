# Animal Classification with Transfer Learning

## Problem Statement
This project addresses the problem of classifying animals that are known to impact crops and livestock. The primary objective is to develop a machine learning model that can accurately identify various animal species based on images, helping farmers mitigate potential damage. Transfer learning is used to fine-tune pre-trained models (VGG16, VGG19, ResNet50) to classify specific animals that commonly interact with agricultural environments.

The animals considered in this classification task include:

- Elephants
- Hippopotamus
- Porcupine
- Warthog
- Baboon

## Dataset
We used a subset of the **ImageNet** dataset for this task. The dataset contains images of the animals mentioned above, with the following characteristics:
- **Relevance**: The dataset focuses on animals that interact with crops and livestock.
- **Size**: The dataset is large enough to provide a meaningful training set.
- **Format**: Images are resized to 224x224 pixels, consistent with the input requirements of pre-trained models such as VGG16, VGG19, and ResNet50.

Data augmentation techniques such as rotation, width/height shifts, horizontal flip, and zoom were applied to increase variability in the training data and prevent overfitting.

## Evaluation Metrics
To assess the performance of the fine-tuned models, the following metrics were used:
- **Accuracy**: The percentage of correct predictions made by the model.
- **Loss**: The model's error as evaluated by the loss function.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

## techniques used
- Early stopping
- L2 regularization

These metrics were chosen to provide a comprehensive evaluation of model performance, especially in dealing with class imbalance and overall accuracy.

## Findings and Discussion

### VGG16
VGG16 provided strong baseline performance due to its deep architecture and large pre-trained weights. However, the model's size and complexity resulted in slower training times. After fine-tuning, VGG16 demonstrated good accuracy, but there were occasional misclassifications between similar bird species.

### ResNet50
ResNet50 showed better performance compared to both VGG models in terms of generalization. ResNet50 achieved faster convergence and slightly better performance in precision and recall.

### VGG19
VGG19, being a deeper variant of VGG16, provided slightly better feature extraction capabilities but required longer training times due to its increased depth. It showed strong performance in terms of accuracy and precision, making it a reliable choice for fine-grained classification tasks like this one.


### Strengths of Transfer Learning
- **Efficiency**: Using pre-trained models allowed me to train with relatively fewer images while still achieving high accuracy.
- **Time-Saving**: Instead of training from scratch, transfer learning significantly reduced the computational cost.
- **Generalization**: The models generalized well to the new task, despite being pre-trained on a broad dataset like ImageNet.

### Limitations
- **Overfitting**: Despite the data augmentation techniques, there is still a risk of overfitting due to the limited number of relevant classes.

## Model Evaluation

| Model      | Accuracy | Loss  | Precision | Recall | F1 Score |
|------------|----------|-------|-----------|--------|----------|
| **VGG16**  | 0.72 |  1.65 | 0.85 | 0.72 | 0.72 |
| **ResNet50**  | 0.33 | 2.36 | 0.16 | 0.33 | 0.21 |
| **VGG19**| 0.77 | 1.56 | 0.86 | 0.77 | 0.77 |

## Conclusion
Transfer learning allowed me to build a highly accurate classification system for animals that could affect crops and livestock in the end interfering with the yield for these farmers. Among the models evaluated, **VGG19** achieved the best overall performance, followed by **VGG16** and **ResNet50**. Future improvements could focus on gathering more relevant data and further fine-tuning for the specific problem at hand.
