# DeepSCI
Tensorflow implementation of paper:DeepSCI: Scalable speckle correlation imaging using physics-enhanced deep learning. We provide the experiment data for a quick demo.
If you find this project useful, we would be grateful if you cite the DeepSCI paper：

Zhiwei Tang, Fei Wang, ZhenFeng Fu, Shanshan Zheng, Ying Jin, and Guohai Situ, "DeepSCI: scalable speckle correlation imaging using physics-enhanced deep learning," Opt. Lett. 48, 2285-2288 (2023).

Abstract
In this Letter, we present a physics-enhanced deep learning approach for speckle correlation imaging (SCI), i.e., DeepSCI. DeepSCI incorporates the theoretical model of SCI into both the training and test stages of a neural network to achieve interpretable data preprocessing and model-driven fine-tuning, allowing the full use of data and physics priors. It can accurately reconstruct the image from the speckle pattern and is highly scalable to both medium perturbations and domain shifts. Our experimental results demonstrate the suitability and effectiveness of DeepSCI for solving the problem of limited generalization generally encountered in datadriven approaches.

Pipeline
![image](https://user-images.githubusercontent.com/129817196/236412525-b0184574-a63a-46c2-9dde-201a460f833d.png)
How to use
Step 1: Configuring required packages

python 3.6

tensorflow 1.9.0

matplotlib 3.1.3

numpy 1.18.1

pillow 7.1.2

Step 2: Change file path and run DeepSCI.py after download and extract the ZIP file.

Results

As you can see in the DeepSCI paper：

Zhiwei Tang, Fei Wang, ZhenFeng Fu, Shanshan Zheng, Ying Jin, and Guohai Situ, "DeepSCI: scalable speckle correlation imaging using physics-enhanced deep learning," Opt. Lett. 48, 2285-2288 (2023).

License
For academic and non-commercial use only.
