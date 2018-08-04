# Notes
We solve lab2s problem using LSTM with CTC loss (see distill post for details). Uses dynamic programming to find the best alignments through marginalization. Inference done through beam search. Model assumes conditional dependence.

ZFNet was the next improvement over AlexNet. Involved just hyperparameter tuning. The paper more popular for deconvolution visualizations.

Next came VGG16 that adopted a uniform architecture applied repeatedly. Conv layers are memory intensive despite fewer params. FC layers require more compute.
GoogLeNet are just as deep as VGG but fewer parameters by removing FC layers. Implements the inception module. Adopts arch that avoids arch search but results in huge depth dimensions. Issue solved by using 1x1 conv layers. 1x1 convs can be used for cross-channel correlations. Inkected classifier loss at earlier layers, which incentivises learning in the earlier layers.

Human performance on ImageNet measured by Andrej Karpathy. 

ResNets used skip connections to avoid depth-related problems with CNN training.DenseNets add more skip connections. ResNext introduces cardinality of a module. 

SENet (2017 SoA) adds a module of pooling+FC to adaptively reweight feature output maps. Similar to the concept of attention. 

SqueezeNet that achieved AlexNet accuracy with 50x fewer params. Uses 1x1 conv layers as the main trick. Interesting for mobile deployments. 

ResNets and Inceptionv4 are probably a nice sweet spot between accuracy/compute cost. 

Overfeat is first network that attempted detection by showing fc-conv equivalency allowing sliding. YOLO and SSD then scaled up the idea by using grid cells. The alternative is region proposal methods (R-CNNs). 

Qs: Has anyone tried pixel-to-pixel heat map generation for detection? Fully Convolutional Networks paper (But how do you upsample? read transpose convolutions)
    Why does YOLO still need non-max suppression? 
    What about the use of other output classifiers than softmax? 
    
Homework:
    Use data augmentation on labs
    Implement Inception v4 and ResNet50 for lab1 and lab3 
    Implement bidirectional LSTMs for lab3
    Read about region proposal, RCNN, Faster RCNN, and Mask RCNN
    Maybe try transfer learning

# Lab 3

In this lab we'll keep working with the EmnistLines dataset.

We will be implementing LSTM model with CTC loss.
CTC loss needs to be implemented kind of strangely in Keras: we need to pass in all required data to compute the loss as inputs to the network (including the true label).
This is an example of a multi-input / multi-output network.

The relevant files to review are `models/line_model_ctc.py`, which shows the batch formatting that needs to happen for the CTC loss to be computed inside of the network, `networks/line_lstm_ctc.py`, which has the network definition.

## Train LSTM model with CTC loss

You need to write code in `networks/line_lstm_ctc.py` to make training work.
Training can be done via

```sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

or the shortcut `tasks/train_lstm_line_predictor.sh`

## Make sure the model is able to predict

You will also need to write some code in `models/line_model_ctc.py` to predict on images.
After that, you should see tests pass when you run

```sh
pipenv run pytest -s text_recognizer/tests/test_line_predictor.py
```

Or you can do `tasks/run_prediction_tests.sh`, which will also run the CharacterModel tests.

## Submitting to Gradescope

Before submitting to Gradescope, add your trained weights to the repo, commit and push your changes:

```sh
git add text_recognizer
git commit -am "my lab3 work"
git push mine master
```

Now go to https://gradescope.com/courses/21098 and click on Lab 3.
Select your fork of the `fsdl-text-recognizer-project` repo, and click Submit.
Don't forget to enter a name for the leaderboard :)

The autograder treats code that is in `lab3/text_recognizer` as your submission, so make sure your code is indeed there.

The autograder should finish in a couple of minutes, and display the results.
Your name will show up in the Leaderboard.

While you wait for the autograder to complete, feel free to try some experiments!

## Things to try

If you have time left over, or want to play around with this later on, you can try using the `line_lstm` network, defined in `text_recognizer/networks/line_lstm.py`.
Code up an encoder-decoder architecture, for example!
