import numpy
import torch

loss = torch.nn.CrossEntropyLoss()

target=torch.from_numpy(numpy.array([0.0, 1.0]))
pred_1=torch.from_numpy(numpy.array([0.0, 1.0]))

print(target)
print(pred_1)
print(torch.nn.functional.cross_entropy(target, pred_1, reduction='none'))

exit()

target=torch.from_numpy(numpy.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
pred_1=torch.from_numpy(numpy.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
pred_2=torch.nn.functional.softmax(torch.from_numpy(numpy.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
pred_3=torch.from_numpy(numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

print(target)
print(loss(pred_1, target).item())
print(loss(pred_2, target).item())
print(loss(pred_3, target).item())