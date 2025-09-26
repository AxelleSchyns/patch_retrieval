from models.arch import *
from training.loss import MarginLoss, ProxyNCA_prob, NormSoftmax, SoftTriple

def loading_weights(model, model_name, weight, device):
        if model_name in ['dino_vit', 'dino_resnet','dino_tiny', "cdpath", "ibot_vits" , "ibot_vitb"]:
                model.load_weights(weight)
                model = model.model

        elif model_name == "byol_light" :
            try:
                model.load_state_dict(torch.load(weight)["state_dict"])
            except:
                model = BYOL(1000).to(device=device)
                model.load_state_dict(torch.load(weight)["state_dict"])

        elif model_name == "ret_ccl":
            pretext_model = torch.load(weight)
            model.fc = nn.Identity()
            model.load_state_dict(pretext_model, strict=True)

        elif model_name in ["phikon", "phikon2", "hoptim", "uni2",  "hoptim1", "virchow2"]: 
            pass

        elif model_name == "ctranspath":
            model.head = nn.Identity()
            model.load_state_dict(torch.load(weight)['model'])

        elif model_name == "uni" or model_name == "virchow2":
            model.load_state_dict(torch.load(weight))
        else:
            try:
                model.load_state_dict(torch.load(weight))
            except Exception as e:
                try:
                    checkpoint = torch.load(weight)
                    print(checkpoint.keys())
                    model.load_state_dict(checkpoint['model_state_dict'])
                    #self.model.load_state_dict(checkpoint)
                except Exception as e:
                    print("Error with the loading of the model's weights: ", e) 
                    print("Exiting...")
                    exit(-1)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

class Model(nn.Module):
    def __init__(self, model_name='resnet', weight='weights', device='cuda:0'):
        super().__init__()
        self.weight = weight
        self.model_name = model_name
        self.device = device

        model_object = load_arch(model_name)
        self.model = model_object[0]
        self.num_features = model_object[1]
        self.model = self.model.to(device=device)

        self.model = loading_weights(self.model, self.model_name, self.weight, self.device)

    def encode(self, image):
        image = image.to(device=self.device, non_blocking=True).reshape(-1, 3, 224, 224)
        with torch.inference_mode():
            if self.model_name == "resnet":
                out = self.model(image)

            elif self.model_name == "deit":
                out = self.model(image)
                out = out.logits

            elif self.model_name == "cdpath":
                image = scale_generator(image, 224, 1, 112, rescale_size=224)
                out = self.model.encode(image)

            elif self.model_name in {"phikon", "phikon2"}:
                outputs = self.model(image)
                out = outputs.last_hidden_state[:, 0, :]

            elif self.model_name in {"hoptim", "hoptim1"} and self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(image)

            elif self.model_name in {"hoptim", "hoptim1"}:  
                out = self.model(image)

            elif self.model_name == "virchow2":
                output = self.model(image)
                class_token = output[:, 0]
                patch_tokens = output[:, 5:]
                out = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

            else:
                out = self.model(image)
                if not isinstance(out, torch.Tensor):
                    out = out.logits

        # 🔹 Normalize for similarity search
        out = F.normalize(out, p=2, dim=-1)
        return out