# deep-learning-challenge
Module 21 Challenge - UWA Data Analytics Bootcamp

#OVERVIEW
The purpose of the analysis contained within this repository is to create a binary classification machine learning model that can predict if organisations funded by Alphabet Soup are successful or not. The prediction is based on features provided in a CSV file.

#RESULTS

Target variable of the model:
 - The IS_SUCCESSFUL column was the target variable for the model. It is a binary variable with 1 indicating success, and 0 indicating failure.
Feature variables of the model: 
 - APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS and ASK_AMT were the features for the model.
Variables that were removed from the input data:
 - The EIN and NAME variables were available in the data but were dropped as features for the model due to the fact that they were for company identification in nature and did not provide any consquential information on company success.
 - Note that there was an attempt to include NAME in the model as an experiment to increase the  accuracy of the model - however this caused the dummies dataframe to inflate to over 19,000  columns and the code would not run successfully.

##Original
hidden_nodes_layer1 = 50
hidden_nodes_layer2 = 25

First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = total_input_features, activation = "relu"))

Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))

Output layer
#Sigmoid used as it is a binary classifier that is applicable to this context
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

fit_model = nn.fit(X_train_scaled,y_train,epochs=50)

268/268 - 0s - 1ms/step - accuracy: 0.7303 - loss: 0.5603
Loss: 0.5602652430534363, Accuracy: 0.7302623987197876

 - The original model used 2 layers (besides the output layer) both with ReLU activation. This was chosen as the input data has features that are non-linear in nature, which ReLU can assist in modeling with its thresholding behaviour.
 - 2 layers provided a good starting point to observe how the model faired with the data.
 - Considering the number of input features was 43 as oberved in the code, 50 nodes was chosen for the first layer as there exists a general rule of thumb that the first hidden layer should have between 2/3 and 3/2 of the input features as nodes.
 - The second layer was designated 25 nodes as in general every subsequent layer should have less nodes than the one preceding it.
 - Sigmoid was used for the output layer for all iterations of the model as the target variable is binary in nature. The sigmoid function maps any real-valued number to the 0,1 range.
 - Overall this model had an accuracy of 73.03% with a loss of 0.56.
 - 50 epochs was chosen as a general starting number that would be further honed.

##Optimisation 1
hidden_nodes_layer1 = 50
hidden_nodes_layer2 = 25

First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = total_input_features, activation = "relu"))

Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))

Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

fit_model = nn.fit(X_train_scaled,y_train,epochs=70)

268/268 - 0s - 2ms/step - accuracy: 0.7335 - loss: 0.5505
Loss: 0.5505353808403015, Accuracy: 0.7335277199745178

 - The first optimisation strategy undertaken was to optimise the number of epochs the model runs.
 - With the original model having 50 epochs, I ran the code with 50, 70 and 80 epochs to gauge an estimate of an optimal epoch number.
 - The best performer of the 3 runs was 70 epochs, with an accuracy of 73.35% and a marginal loss improvement.

##Optimisation 2
hidden_nodes_layer1 = 60
hidden_nodes_layer2 = 35

First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = total_input_features, activation = "relu"))

Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))

Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

fit_model = nn.fit(X_train_scaled,y_train,epochs=70)

268/268 - 0s - 2ms/step - accuracy: 0.7403 - loss: 0.5494
Loss: 0.5494025945663452, Accuracy: 0.7402915358543396

 - The second optimisation strategy undertaken was to improve the number of nodes in the model layers.
 - First the node count was reduced, however this instantly decreased model accuracy consistently.
 - Increasing the node count by a small amount created the most improvement with 60 and 35 nodes for the first and second hidden layer respectively increasing model accuracy to 74.03% with a marginal loss improvement.

##Optimisation 3
hidden_nodes_layer1 = 60
hidden_nodes_layer2 = 35
hidden_nodes_layer3 = 15

nn = tf.keras.models.Sequential()

First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim = total_input_features, activation = "relu"))

Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation = "relu"))

Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation = "sigmoid"))

Output layer
#Sigmoid used as it is a binary classifier that is applicable to this context
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

fit_model = nn.fit(X_train_scaled,y_train,epochs=70)

268/268 - 0s - 2ms/step - accuracy: 0.7366 - loss: 0.5488
Loss: 0.5488268136978149, Accuracy: 0.7365597486495972

 - The third strategy for improvement was adding another hidden layer and customising its nodes + activation function to generate better accuracy.
 - Several experimental runs were commenced with different activation functions, sigmoid appeared to create the highest accuracy yields and was chosen.
 - Considering the preceding layer had 35 nodes, 17 nodes was originally chosen as a middle ground. Tuning this down to 15 nodes created the highest accuracy for this model iteration.
 - Unfortunately adding a new layer struggled to increase accuracy in any of its experimental runs. It had a final accuracy of 73.66% but had a marginal improvement in loss.

#SUMMARY
Overall the best iteration of the model was optimisation 2 which contained 2 hidden layers with ReLU activation at 60 and 35 nodes and 70 epochs. 

A random forest model would make a good alternative in this context as it is adept at capturing non-linear relationships which this data appears to contain a lot of. Considering that each decision tree in this type of model is trained independently on random subsets of data and features, it makes it resistant to overfitting which could vastly improve accuracy on this data.