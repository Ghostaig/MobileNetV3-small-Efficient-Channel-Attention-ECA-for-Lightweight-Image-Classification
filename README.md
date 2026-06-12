# MobileNetV3-small-Efficient-Channel-Attention-ECA-for-Lightweight-Image-Classification
This project enhances the lightweight MobileNetV3-small image classification model using Efficient Channel Attention (ECA). A baseline and an ECA-enhanced version were compared to improve classification accuracy while maintaining low computational requirements for mobile and embedded device deployment.

Chapter One

1.1 Introduction
Deep learning models have achieved very high performance in image classification tasks, but many of them are too large and computationally expensive for mobile and embedded systems. MobileNetV3-small was developed as a lightweight model that can run efficiently on devices with limited computing power. This project improves MobileNetV3-small using Efficient Channel Attention (ECA). Two models were compared: a baseline MobileNetV3-small model and an ECA-enhanced version. The aim was to improve accuracy while still keeping the model lightweight and suitable for deployment.


1.2 Problem Statement
Many transfer learning systems freeze most of the pretrained layers and only retrain the classifier section. Although this reduces training cost, it can limit the model’s ability to learn important features from new datasets. Also, some attention mechanisms improve accuracy but make the model heavier. This project investigates whether ECA can improve MobileNetV3-small without significantly increasing computational cost.


1.3 Related Work
Howard et al (2019), introduced MobileNetV3 by combining Neural Architecture Search, NetAdapt optimization, and Squeeze-and-Excitation modules to produce efficient mobile CNN architectures.
Wang et al. later proposed ECA-Net (2020), replaced expensive dimensionality reduction operations with lightweight local cross-channel interactions using 1D
Zhuolei et al and hangyu et al (2025), Improving network structure for efficient classification network based on mobilenetV3, they proposed an improved mobilenetV3 that integrates Efficient Channel Attention (ECA), redesigned bottleneck layers, trivial argument and SiLU activation to improve lightweight image classification performance. 
These transfer learning studies further demonstrate that differential learning rates, label smoothing, and adaptive schedulers significantly improve convergence stability on relatively small datasets.

The methodology implemented in this project combines these modern optimization techniques. ECA modules are inserted at strategic feature extraction stages while the original heavy classifier head is replaced with a simplified lightweight projection layer. This creates a balance between computational efficiency and feature discrimination.


Chapter Two


2.1 Methodology
The pipeline follows a two-model comparative design:
A. Preprocessing: The Images resized to 256×256×3, normalized using ImageNet statistics, Training set augmented with horizontal flip, ±15° rotation, and mild color jitter. Validation set uses deterministic resizing and center cropping.
B. Model 1 (Baseline model): uses the pretrained mobilenetv3 small (ImageNet).The First two feature blocks were frozen to preserve generic edge/texture features. Remaining blocks and classifier head fine-tuned.
C. Model 2 (ECA-Enhanced & Lighter): This Integrates ECA modules after feature blocks 4 and 7. The original classifier head was replaced with Dropout (0.25), Linear (576, num_classes), reducing to 590K parameters. ECA applies global average pooling followed by adaptive 1D convolution (k=3) to recalibrate channel responses without dimensionality reduction.
D. Training Strategy: Differential learning rates of (backbone: 2.5e-5, head/ECA: 2.5e-4), AdamW optimizer (weight decay: 1e-4), label smoothing (0.05), and ReduceLROnPlateau scheduler. With an Early stopping (patience=10) to prevent overfitting.


2.2 Dataset
Dataset 1 comprises 4 distinct classes with 9,888 total images. The dataset was naturally pre-split using a stratified 80:20 ratio, yielding 7,910 images for training and 1,978 images as validation samples, preserving class distribution. All images were processed as 256×256 RGB tensors and normalized using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]. Class weights were computed to mitigate minor sampling imbalances. Light augmentation was applied exclusively during training to improve generalization without distorting validation distributions.


2.3 Experiments
Training was executed on Google Colab (NVIDIA T4 GPU, PyTorch 2.0+). A Batch size of 32 was used. Both models were trained for up to 50 epochs. Model 1 achieved a validation accuracy of 65.32% with stable loss curves, and Model 2 utilized the ECA-enhanced architecture with differential LRs and a streamlined classifier head. Evaluation metrics (Accuracy, Precision, Recall, and F1-Score) were computed using weighted averages. FLOPs and parameter counts were measured using the thop profiler. Confusion matrices and accuracy-vs-epoch plots were generated post-training. All hyper parameters and architectural decisions were logged for reproducibility.


2.4 Improvement
Model 2 showed clear improvement over the baseline model. Trainable parameters were reduced from 876,108 to 652,340, while validation accuracy improved from 65.32% to 69.42%. The ECA modules improved feature selection, while the lightweight classifier reduced unnecessary computational complexity.
Model 2 is 25.5% lighter in trainable parameters while achieving a +4.1% absolute accuracy improvement. The simplified classifier head reduces over parameterization, while ECA’s zero-parameter attention compensates for reduced capacity by focusing on discriminative mid-level features. This confirms that strategic architectural pruning combined with efficient attention yields superior efficiency-accuracy trade-offs.
Performance Comparison
Metric        Improvement     Accuracy    Trainable Params    F1-Score
Model 1                       65.32%       876,108             65.44%
Model 2                       69.42%       652,340             70.85%
Improvement                   +4.1%        -25.5%              +5.41%



Chapter Three


3.1 Results and Discussion
1. Parameter & FLOPs Analysis: Model 1 uses 0.079 GFLOPs while Model 2 uses 0.074 GFLOPs. The 6.3% FLOP reduction aligns with the simplified head and ECA’s lightweight 1D convolutions.
2. Error Analysis: Confusion matrices reveal that Model 2 significantly reduces misclassifications between visually similar classes. ECA’s channel recalibration enhances feature separation without increasing computational burden.
3. Discussion: The success of Model 2 validates the hypothesis that heavy classifier heads in transfer learning often bottleneck performance. Replacing them with direct projections, combined with ECA’s parameter-free attention, forces the network to learn more compact, discriminative representations. Label smoothing and differential LRs further stabilized training. The Accuracy target is was almost met, and the model remains significantly lighter than the baseline.


3.2 Conclusion
MobileNetV3-small transfer learning with ECA integration was successfully integrated on Dataset 1. Model 1 established a robust baseline (65.32%), while Model 2 achieved 69.42% accuracy while reducing trainable parameters by 25.5%. The integration of ECA and classifier head simplification proves highly effective for lightweight, high-accuracy classification. All project requirements including increased accuracy, parameter reduction, comprehensive evaluation, and improvement calculation are met. Future work will explore Test-Time Augmentation (TTA) for inference boosting, knowledge distillation for further compression, and quantization-aware training for edge deployment









References
Howard, A., Sandler, M., Chu, G., et al. (2019). Searching for MobileNetV3. Proceedings of ICCV.

Wang, Q., Wu, B., Zhu, P., et al. (2020). ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. Proceedings of CVPR.

Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. ICLR.

Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR.

Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

[Lyu, Y., et al. (2020). thop: PyTorch-OpCounter. GitHub Repository.
