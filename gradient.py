import torch


class Gradient(torch.nn.Module):
    r'''
    Compute the first-order local gradient
    '''

    def __init__(self) -> None:
        super().__init__()

        self.d_x = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.d_y = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.update_weight()

    def update_weight(self):
        sobel_filter = torch.FloatTensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        kernel_dx = sobel_filter.unsqueeze(0).unsqueeze(0)
        kernel_dy = sobel_filter.transpose(1, 0).unsqueeze(0).unsqueeze(0)

        self.d_x.weight = torch.nn.Parameter(kernel_dx).requires_grad_(False)
        self.d_y.weight = torch.nn.Parameter(kernel_dy).requires_grad_(False)

    def forward(self, img):
        batch_size = img.shape[0]
        img_aux = img.reshape(-1, img.shape[-2], img.shape[-1])
        img_aux.unsqueeze_(1)
        grad = self.d_x(img_aux).pow(2)
        grad += self.d_y(img_aux).pow(2)
        grad.sqrt_()
        grad.squeeze_(1)
        grad = grad.reshape(
            batch_size, -1, img_aux.shape[-2], img_aux.shape[-1])
        return grad
