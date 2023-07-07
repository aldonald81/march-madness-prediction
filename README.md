# NCAA March Madness Prediciton
I created a k-nearest neighbors algorithm-based model to classify tournament teams as “x-round” caliber teams to predict the NCAA March Madness bracket. My thought process for using this type of model was to predict a team's
success in this year's tournament based off that of the most comparable historical teams with respect to stats and attributes. By using the k-closest teams from previous years, I average their success and use that to define
the tier for this year's team. When predicting the bracket, I move ahead teams that have a higher tier. I fine-tuned model hyperparameters using cross-validation, resulting in a 96th percentile score in the 2022 ESPN bracket challenge.
Although the success will vary, I think this is a unique approach to attempting to solve some of the madness.

