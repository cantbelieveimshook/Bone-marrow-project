from UNET import *

# loading all image paths into list
# can be used for any any amount of nested folders (hopefully, as long as it works properly)

mainfolder = '/content/drive/Shareddrives/Bone Marrow Classification'
labels = ['ABE', 'ART', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC', 'KSC', 'LYI', 'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM', 'PMO']

# loads all image paths from any folder, works with any combination of folders/degrees of nested folders
# currently modified to only add the first five image paths from each folder
# labels are currently unfortunately hardcoded because idk how else to get the labels from the images
def loadimagepaths(folder, images = []):
  i = 0
  for data in os.listdir(folder):
    subfolder = folder + '/' + data
    if (folder + '/.ipynb_checkpoints') in os.listdir(folder):
      os.listdir(folder).remove(folder + '/.ipynb_checkpoints')
    if data[-4:] == '.jpg':
      images.append(subfolder)
      label = subfolder[55:58]
      i += 1
    else: 
      loadimagepaths(subfolder, images)
    if i >= 10:
      break
  return images

all_image_paths = loadimagepaths('/content/drive/Shareddrives/Bone Marrow Classification')

# writing image paths and labels onto csv file
f = open('./dataset_info.csv', 'a')
f.write("Path, Label\n")

for path in all_image_paths:
  label = path[55:58]
  f.write(path + ',' + label + '\n')

f.close()

dataset=BoneMarrowDataset('dataset_info.csv')

train, test = train_test_split(dataset, test_size = 0.1, random_state = 1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
labels = ['ABE', 'ART', 'BLA', 'EBO', 'FGC', 'EOS', 'HAC', 'KSC', 'LYI', 'LYT', 'NGS', 'NGB', 'NIF', 'MYB', 'MON', 'MMZ', 'PEB', 'PLM', 'PMO', 'OTH']
le = preprocessing.LabelEncoder()
encoder = preprocessing.OneHotEncoder()
labels = le.fit(labels)
# encoder.fit(labels.reshape(-1, 1))
# labels = encoder.transform(labels.reshape(-1, 1))

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label = data
        inputs = inputs.float()
        label = le.transform([label])
        encoder.fit(label.reshape(-1, 1))
        label = encoder.transform(label.reshape(-1, 1))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, torch.as_tensor(label))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 1:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('B)')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test:
        inputs, label = data
        inputs = inputs.float()
        label = torch.as_tensor(le.transform([label]))
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in labels}
total_pred = {classname: 0 for classname in labels}

# again no gradients needed
with torch.no_grad():
    for data in test:
        inputs, classifications = data
        inputs = inputs.float()
        classifications = torch.as_tensor(le.transform([classifications]))
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(classifications, predictions):
            if label == prediction:
                correct_pred[labels[label]] += 1
            total_pred[labels[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
  if total_pred[classname] != 0:
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
  else:
    print(f'No predictions for this class: {classname}')

