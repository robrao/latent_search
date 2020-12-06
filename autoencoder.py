import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision.utils.tensorboard import SummaryWriter

# NOTE: Directory to store reconstructed images
# to see progress during training.
if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

# NOTE: Reconstruct image from output layer
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 12), nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), nn.ReLU(True),
            nn.Linear(12, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28), nn.Tanh() # TODO: Sigmoid vs Tanh?
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    dataset = MNIST('./data', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Moves all model parameters and buffer to the GPU
    # model = Autoencoder().cuda()
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            # img = Variable(img).cuda()
            img = Variable(img)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))
    
    torch.save(model.state_dict(), './autoencoder.pth')
