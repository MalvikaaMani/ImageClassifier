<h1> Project Overview </h1>
This project is an interactive **image classification web app** built with Streamlit. It classifies uploaded images as either a *cat* or a *dog* with high accuracy using a Convolutional Neural Network trained on the popular Dogs vs Cats dataset. It uses <b> MobileNetV2 </b> for building the model. 
<br>
<p><b> Live Demo </b> </p>
<br>
[Click here to try it on Streamlit](https://imageclassifier-fjtnkjrdhcfpwaqsjdeejw.streamlit.app/)
<br>
<h2> Features </h2>
<ul>
<li>Upload any image of a cat or dog</li>
<li>Classify with a single click</li> 
<li>Instant results with confidence scores</li> 
<li>Engaging and user-friendly interface</li>
<li> Deployed on Streamlit Cloud </li>
<li>Built using TensorFlow and Keras</li>


<h1> Dataset Used </h1>
The famous cats vs dogs dataset available on Kaggle.
<br>
[Download the dataset from here](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
<ul>
<li> 2 classes </li>
<ul>
<li> cats </li>
<li> dogs </li>
</ul>
</ul>

<h1>Results</h1>
The model was trained on a binary classification problem (cats vs. dogs). The performance metrics are summarized below:
<ul>
<li><b> Loss curve </b>:As shown in the loss plot (Loss_plot.png), the training and validation loss decreased steadily over 5 epochs, with some slight divergence indicating mild overfitting. The final validation loss stabilized around 0.063.</li>
<li><b> Accuracy Curves </b>:From the accuracy plot (Accuracy_plot.png), the training accuracy reached approximately 98%, while validation accuracy remained around 97.5%, suggesting reasonable generalization on the validation set.</li>
<li> <b>Classification Report </b>:As shown in the classification report (Classification_report.png), the model achieved:
<ul>
<li>Precision: 0.50 for both classes</li>
<li>Recall: 0.49 for cats, 0.50 for dogs</li>
<li>F1-score: 0.50 for both classes</li>
</ul>
</ul>
