import torch
import torch.nn.functional as F

# TODO - none of this is implemented/tested, it's possibly going to be useful for any affine image transforms

def __make_grid(self, xyt):
    grid = torch.zeros(xyt.shape[0], 2, 3).to(self.device)
    grid[:, 0, 0] = torch.cos(xyt[:, 0])
    grid[:, 1, 1] = torch.cos(xyt[:, 0])
    grid[:, 0, 1] = torch.sin(xyt[:, 0])
    grid[:, 1, 0] = -torch.sin(xyt[:, 0])
    grid[:, :, 2] = xyt[:, 1:]
    return grid


def make_grid(self, xyt):
    return self.__make_grid(xyt)


def interpolate(self, x, grid):
    if grid.shape[1] == 3:
        grid = grid[:, :2, :]
    grid = F.affine_grid(grid, x.size())
    if grid.dtype == torch.half:
        x = F.grid_sample(x.half(), grid).float()
    else:
        x = F.grid_sample(x, grid)
    return x


def move_image(self, image, xyt):
    grid = self.make_grid(xyt)
    image = self.interpolate(image, grid)
    return image


def compose_se2(tens, multiplier=1):
    tens = tens.cpu().squeeze().detach().numpy()
    tens[1] *= multiplier
    tens[2] *= multiplier
    rot = compose_se2(tens[0], 0, 0)
    trasl = compose_se2(0, tens[1], tens[2])
    tranf = rot @ trasl
    return tranf
