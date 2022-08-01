install.packages("readxl")
install.packages("torch")
library(torch)
library(readxl)
library(data.table)
library(caret)

if (cuda_is_available()) {
  device <- torch_device("cuda")
} else {
  device <- torch_device("cpu")
}

#1. Load the dataset: Dry beans ------
dt <- read_excel("~/Exercise3 - Clustering K-means - Orschanski/Dry_Bean_mod.xlsx")
dt <- data.table(dt)

dt[, Class := as.factor(Class)] 
dt[, .(n_obs = .N, share = .N / nrow(dt)), Class] 

  #6 labels in the output Class

#2. It's ordered so i should shuffle it for later modeling
set.seed(123)
dt<-dt[sample(1:nrow(dt))]


#3. Split the dataset
set.seed(123)
index <- createDataPartition(dt[,Class], p = 0.8, list = F, times = 1) 
  #Stratified splitting in order to keep the same proportions of the classes in the datasets

training <- dt[index] #training set is the 80% from the dataset: 10473 observations
test <- dt[!index]#test set is the 20%: 2616 is the 20% of 13089 (observations)

#3. Adapt the data for conversion
x_train <- as.matrix(training[, 1:16])
typeof(training[,17]) #i cant convert a list into a integer, so I will do it this way:
y_train <- as.integer(training$Class)

x_test <- as.matrix(test[, 1:16])
y_test <- as.integer(test$Class)



#4. Convert to tensors (multi dimensional vector)
x_train_tensor <- torch_tensor(x_train, dtype = torch_float())
y_train_tensor <- torch_tensor(y_train, dtype = torch_long())

x_test_tensor <- torch_tensor(x_test, dtype = torch_float())
y_test_tensor <- torch_tensor(y_test, dtype = torch_long())

#5. Create the NN model---
  #5.1 One option is:
model1 <- nn_sequential(
  # layer 1 (16 inputs and 32 outputs)
  nn_linear(16, 32), nn_relu(),
  # layer 2 (32 inputs and 64 outputs)
  nn_linear(32, 64), nn_relu(),
  # layer 3 (64 inputs and 6 outputs)
  nn_linear(64, 6), nn_softmax(dim = 2)
)
  #The first activation function is ReLu: rectified linear.  the  bias determine the threshold from which the neuron will trigger  
  #The last activation function is SoftMax: rescales the inputs


#6. Define cost function and optimizer
criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(model1$parameters, lr = 0.005)
  #lr = learning rate
  #optimizer = adam
  #the parameters of the model are the weights and the bias

#7. Define the number of epochs
  #determines when to stop
  #this depends on when the model converges (how the loss performs)
  #we will see that the loss decreases until a point in which it either stays stable or increases again
epochs <- 200

for (i in 1:epochs) {
  optimizer$zero_grad()
  
  # Forward pass
  y_pred_tensor <- model1(x_train_tensor) #compute the output
  
  # Compute loss
  loss <- criterion(y_pred_tensor, y_train_tensor)
  loss$backward()
  
  # take a step in the opposite direction
  optimizer$step()
  
  if (i %% 10 == 0) { #i will print every 10 iterations
    winners <- y_pred_tensor$argmax(dim = 2)
    corrects <- winners == y_train_tensor
    accuracy <- corrects$sum()$item() / y_train_tensor$size()
    cat("Epoch:", i,
        "Loss", loss$item(),
        "Accuracy", accuracy, "\n")
  }
}

  #The results stay the same within the iterations: Epoch: 200 Loss 1.888718 Accuracy 0.1548744 
  
# Check on the test set
y_pred_tensor <- model1(x_test_tensor)
y_pred <- as.array(y_pred_tensor$argmax(dim = 2))

print(table(y_pred, y_test))
#it is just predicting the class 5
cat(" Accuracy: ", sum(y_pred == y_test) / length(y_pred), "\n")
  #The accuracy is extremely low: 0.15

#I am having a problem because:
#If I run this code again I have different results: Accuracy stays in 0.1010217 and predicts class 1
#Also, another couple of results are: Accuracy staying in 0.1245106 and predicts class 2



#8. I will try with another model:------------------------------------------------

model2 <- nn_sequential(
  # layer 1 (16 inputs and 24 outputs)
  nn_linear(16, 24), nn_relu(),
  # layer 2 (24 inputs and 32 outputs)
  nn_linear(24, 32), nn_relu(),
  # layer 3 (32 inputs and 6 outputs)
  nn_linear(32, 6), nn_softmax(dim = 2)
)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_sgd(model2$parameters, lr = 0.01) #higher learning rate
  #using a small learning rate might mean we will get stuck in a minimum that is not the minimum we are looking for. We will be stucked in a loss that is high
  #using a large learning rate might lead to diverge so we would never find the minimum.
  #optimizer : SGD

epochs <- 400

for (i in 1:epochs) {
  optimizer$zero_grad()
  
  # Forward pass
  y_pred_tensor <- model2(x_train_tensor) #compute the output
  
  # Compute loss
  loss <- criterion(y_pred_tensor, y_train_tensor)
  loss$backward()
  
  # take a step in the opposite direction
  optimizer$step()
  
  if (i %% 10 == 0) { #i will print every 10 iterations
    winners <- y_pred_tensor$argmax(dim = 2)
    corrects <- winners == y_train_tensor
    accuracy <- corrects$sum()$item() / y_train_tensor$size()
    cat("Epoch:", i,
        "Loss", loss$item(),
        "Accuracy", accuracy, "\n")
  }
}
  #the results of accuracy stay the same: Accuracy 0.270887 
  
# Check on the test set
y_pred_tensor <- model2(x_test_tensor)
y_pred <- as.array(y_pred_tensor$argmax(dim = 2))

print(table(y_pred, y_test))
  #it is only predicting the class 3

cat(" Accuracy: ", sum(y_pred == y_test) / length(y_pred), "\n")
#The accuracy is higher: 0.27

#I an having the same problem:
#If I run this code again I have different results: Loss 1.842217 Accuracy 0.201375  and predict class 6


#9. I will try with another model:--------------------------------------------------------

model3 <- nn_sequential(
  # layer 1 (16 inputs and 32 outputs)
  nn_linear(16, 32), nn_relu(),
  # layer 2 (32 inputs and 6 outputs)
  nn_linear(32, 6), nn_relu()
  
)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_sgd(model3$parameters, lr = 0.01) 

epochs <- 400 #number of steps we are going backwards

for (i in 1:epochs) {
  optimizer$zero_grad()
  
  # Forward pass
  y_pred_tensor <- model3(x_train_tensor) #compute the output
  
  # Compute loss
  loss <- criterion(y_pred_tensor, y_train_tensor)
  loss$backward()
  
  # take a step in the opposite direction
  optimizer$step()
  
  if (i %% 10 == 0) { #i will print every 10 iterations
    winners <- y_pred_tensor$argmax(dim = 2)
    corrects <- winners == y_train_tensor
    accuracy <- corrects$sum()$item() / y_train_tensor$size()
    cat("Epoch:", i,
        "Loss", loss$item(),
        "Accuracy", accuracy, "\n")
  }
}

#The Loss decreases from 1.797262 to 1.759069 
#The Accuracy stays the same: 0.270887 


# Check on the test set
y_pred_tensor <- model3(x_test_tensor)
y_pred <- as.array(y_pred_tensor$argmax(dim = 2))

print(table(y_pred, y_test))
#it just predict on the class 3
cat(" Accuracy: ", sum(y_pred == y_test) / length(y_pred), "\n")
#The accuracy is still low: 0.27 


#If I run this code another time:
#The loss decreases and the accuracy increase in epoch 100. Then, the loss still decreases and the accuracy stays the same
  #Epoch: 90 Loss 1.78682 Accuracy 0.1245106 
  #Epoch: 100 Loss 1.785659 Accuracy 0.270887 


#10. I will try with another model:--------------------------------------------------------

model4 <- nn_sequential(
  # layer 1 (16 inputs and 32 outputs)
  nn_linear(16, 32), nn_relu(),
  # layer 2 (32 inputs and 64 outputs)
  nn_linear(32, 64), nn_relu(),
  # layer 3 (64 inputs and 6 outputs)
  nn_linear(64, 6), nn_relu()
  
)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_sgd(model4$parameters, lr = 0.01) 

epochs <- 400 #number of steps we are going backwards

for (i in 1:epochs) {
  optimizer$zero_grad()
  
  # Forward pass
  y_pred_tensor <- model4(x_train_tensor) #compute the output
  
  # Compute loss
  loss <- criterion(y_pred_tensor, y_train_tensor)
  loss$backward()
  
  # take a step in the opposite direction
  optimizer$step()
  
  if (i %% 10 == 0) { #i will print every 10 iterations
    winners <- y_pred_tensor$argmax(dim = 2)
    corrects <- winners == y_train_tensor
    accuracy <- corrects$sum()$item() / y_train_tensor$size()
    cat("Epoch:", i,
        "Loss", loss$item(),
        "Accuracy", accuracy, "\n")
  }
}

#The loss decreases and the accuracy increases in epoch 40: 
  #Epoch: 30 Loss 1.788138 Accuracy 0.1245106 
  #Epoch: 40 Loss 1.784709 Accuracy 0.270887
#Epoch: 90 Loss 1.791423 Accuracy 0.1473312 
#Epoch: 100 Loss 1.791227 Accuracy 0.201375

# Check on the test set
y_pred_tensor <- model4(x_test_tensor)
y_pred <- as.array(y_pred_tensor$argmax(dim = 2))

print(table(y_pred, y_test))
#it just predict the class 3 

cat(" Accuracy: ", sum(y_pred == y_test) / length(y_pred), "\n")
#The accuracy is: 0.20


#If I run this code another time I get an increase in accuracy and a decrease in loss in epoch 80
#Epoch: 70 Loss 1.781085 Accuracy 0.1548744 
#Epoch: 80 Loss 1.779873 Accuracy 0.270887 
#After this the loss decreases to 1.761304 and the accuracy stays the same

#11. I will try with another model---------------------------------------------------------------------4
model5 <- nn_sequential(
  # layer 1 (16 inputs and 24 outputs)
  nn_linear(16, 64), nn_relu(),
  # layer 2 (24 inputs and 32 outputs)
  nn_linear(64, 32), nn_relu(),
  # layer 3 (24 inputs and 32 outputs)
  nn_linear(32, 6), nn_relu()
  
)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_sgd(model5$parameters, lr = 0.005) 

epochs <- 400 #number of steps we are going backwards

for (i in 1:epochs) {
  optimizer$zero_grad()
  
  # Forward pass
  y_pred_tensor <- model5(x_train_tensor) #compute the output
  
  # Compute loss
  loss <- criterion(y_pred_tensor, y_train_tensor)
  loss$backward()
  
  # take a step in the opposite direction
  optimizer$step()
  
  if (i %% 10 == 0) { #i will print every 10 iterations
    winners <- y_pred_tensor$argmax(dim = 2)
    corrects <- winners == y_train_tensor
    accuracy <- corrects$sum()$item() / y_train_tensor$size()
    cat("Epoch:", i,
        "Loss", loss$item(),
        "Accuracy", accuracy, "\n")
  }
}

#Epoch: 10 Loss 1211.623 Accuracy 0.270887 
#Epoch: 20 Loss 1.765099 Accuracy 0.201375
#Epoch: 130 Loss 1.753775 Accuracy 0.201375 
#Epoch: 140 Loss 1.753576 Accuracy 0.270887 

#After this the values of accuracy stay the same

# Check on the test set
y_pred_tensor <- model5(x_test_tensor)
y_pred <- as.array(y_pred_tensor$argmax(dim = 2))

print(table(y_pred, y_test))
#just predicts class 3
cat(" Accuracy: ", sum(y_pred == y_test) / length(y_pred), "\n")
#The accuracy is the same as the previous case: 0.27

#If I run this code again:
#Epoch: 210 Loss 1.777229 Accuracy 0.1245106 
#Epoch: 220 Loss 1.776556 Accuracy 0.270887





