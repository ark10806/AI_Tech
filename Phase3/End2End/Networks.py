import timm

n_classes = 18

def Efficientnet_b2():
    model = timm.create_model("efficientnet_b2", pretrained=True)
    terminal = model.classifier
    terminal.out_features=n_classes
    stdv = 1 / terminal.in_features ** 0.5
    terminal.bias.data.uniform_(-stdv, stdv)
    for param in model.parameters():
        param.requires_grad = True
    
    return model