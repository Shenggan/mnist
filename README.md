# CS420 Project: MNIST Handwritten-Digits Recognition

In this project, we investigate the mainstream techniques used in hand-written digit recognition task and propose semi-supervise(self-supervise) approaches in dealing with digits recognition with only a small fraction of labeled images. The dataset given for this project is based on Mixed National Institute of Standards and Technology database (MNIST) with transformations like adding noise and moving digit center. Traditional machine learning models like softmax regression, support vector machine (SVM), k-nearest neighbors(k-NN) as well as more advanced deep learning approaches like VGG, Resnet are also included. Some models can achieve nearly perfect performance(99%+ accuracy). Finally, we present our new method which shows comparable performance to most state-of-art models while only use negligible samples.

![mnist](./img/sample.png)

### Usage

1. Download and extract the dataset:

    ```shell
    ./dataset.sh
    ```

2. Traditional Machine Learning:

    You can run *SVM* model just like below. The `model_type` can be `svm`, `nusvm`, `knn`, `sr`. And you can see more optioal with `python main.py --help`

    ```shell
    cd traditional_machine_learning/
    python main.py --model_type svm
    # if you want to use PCA and knn
    python main.py --model_type knn --pca
    ```

3. ResNet and VGG:

    ```shell
    cd resnet_and_vgg/
    # if you want to try vgg
    python main_vgg.py
    # if you want to try resnet
    python main_resnet.py
    ```

4. Virtual Adersarial Training

    You can see more details about VAT in [./virtual_adversarial_training/README.md](./virtual_adversarial_training/README.md).

    ```shell
    cd virtual_adversarial_training
    python train_sup.py --cost_type=VAT_finite_diff --epsilon=2.1 --layer_sizes=784-1200-600-300-150-10 --save_filename=<filename>
    python train_semisup.py --cost_type=VAT_finite_diff --epsilon=0.3 --layer_sizes=784-1200-1200-10 --num_labeled_samples=100 --save_filename=<filename>
    ```

5. Deep Generate Model:

    ```shell
    cd deep_generate_model/
    python main.py
    ```

### Results

1. Traditional Machine Learning

    | Model  | Training Accuracy | Testing Accuracy   | 
    | :--:               | :--:  | :--:   | 
    | Softmax Regression | 32 | 29  |
    | KNN                | - | 89  | 
    | SVM                | **94** | **93**  | 
    | MLP                | 92 | 87  | 

2. Deep Learing Method

    | TrainSet Size | VGG   | ResNet | DGM   | VAT   |
    | :--:          | :--:  | :--:   | :--:  | :--:  |
    | 100           | 50.74 | 51.52  | 90.09 | **98.07** |
    | 1000          | 92.23 | 95.81  | **95.98** | -     |
    | 10000         | 98.90 | **98.93**  | 98.65 | -     |
    | 60000(ALL)    | 99.64 | **99.70**  | 99.13 | 99.32 |

