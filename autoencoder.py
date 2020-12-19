import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

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
    writer = SummaryWriter()
    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Moves all model parameters and buffer to the GPU
    # model = Autoencoder().cuda()
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    for epoch in range(num_epochs):
        loss_val = float("Inf")
        images = None
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            # img = Variable(img).cuda()
            img = Variable(img)

            images = img
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            loss_val = float(loss)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)
        writer.add_scalar('Loss', loss_val, epoch)

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss_val))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    writer.close()
    torch.save(model.state_dict(), './autoencoder.pth')
