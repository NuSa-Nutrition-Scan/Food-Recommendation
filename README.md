<h1>Food Recommendation</h1>
<p>
This repository contains code to provide food recommendations to users according to the desired calories and how many times the user wants to eat each day. This project aims to make it easier for users to choose what to eat each day, and of course the recommended meals are in accordance with the desired calorie needs.

Through our research, we found an approach to find a solution to the problem. We used the approach of building with Ranking and Retrieval Model with Query and Candidate tower model.
</p>

<h2>Overview</h2>

<p>Program flow is as follows :</p>
<ul>
  <li>--.</li>
  <li>--.</li>
  <li>--.</li>
</ul>

<h2>How to Replicate Process</h2>
<ol>
  <li>First clone this repository using comment "git clone git@github.com:NuSa-Nutrition-Scan/Food-Recommendation.git"</li>
  <li>Install requirements needed</li>
  <li>If you want to do training data, you can access it at Recommendation System.ipynb</li>
  <li>If you want to do a deployment to run the script, you can use Demo-deploy-with-gradio.py</li>
</ol>

<h2>Directory Structure</h2>
<ol>
  <li>dataset</li>
  This folder contains dataset that we use to training food recommendation model with csv format
  <li>saved_model</li>
  This folder contains weight of parameter from food recommendation model that we already train from our dataset
  <li>Demo-deploy-with-gradio.py</li>
  This python script contains script for using our food recommendation model with the help of gradio package
  <li>Recommendation System.ipynb</li>
  This is jupyter notebook that we use for training our food recommendation model
  <li>.pkl file</li>
  The file that using .pkl name is the pickle form of our dataset that we use for training food recommendation model
  <li>food_recom_model.py</li>
  This python script constain script to define our food recommendation model
  <li>requirements.txt</li>
  List of python package that we use to training and deploying our food recommendation model
</ol>








