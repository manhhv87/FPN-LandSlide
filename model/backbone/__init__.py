from model.backbone import resnet


def build_backbone(back_bone, pretrained=True):
    if back_bone == "resnet101":
        return resnet.ResNet101(pretrained=pretrained)
