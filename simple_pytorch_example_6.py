#!/usr/bin/env python

from imports import *
from going_modular.going_modular import data_setup, engine, utils
from my_utils import set_seeds

IMG_SIZE = 224 
BATCH_SIZE = 32 

class PatchEmbedding(nn.Module):
  def __init__(self,
               in_channels:int=3,
               patch_size:int=16,
               embedding_dim:int=768): 
    super().__init__()

    self.patch_size = patch_size
  
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)
    
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)
    
  def forward(self, x):
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

    x_patched = self.patcher(x) 
    x_flattened = self.flatten(x_patched)
    return x_flattened.permute(0, 2, 1)

class MultiHeadSelfAttentionBlock(nn.Module): 
  """MultiHeadSelfAttentionBlock creates a multi-head self-attention block ("MSA block" for short). """ 
  def __init__(self, 
               embedding_dim:int=768, 
               num_heads:int=12, 
               attn_dropout:int=0):
    super().__init__()
    
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                num_heads=num_heads,
                                                dropout=attn_dropout, 
                                                batch_first=True) 
  
  def forward(self, x):
    x = self.layer_norm(x)
    attn_output, _ = self.multihead_attn(query=x,
                                         key=x,
                                         value=x,
                                         need_weights=False)
    return attn_output

class MLPBlock(nn.Module):
  def __init__(self,
               embedding_dim:int=768,
               mlp_size:int=3072,
               dropout:int=0.1):
    super().__init__()
    
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embedding_dim),
        nn.Dropout(p=dropout) 
    )
  
  def forward(self, x):
    x = self.layer_norm(x) 
    x = self.mlp(x)
    return x

class TransformerEncoderBlock(nn.Module):
  def __init__(self,
               embedding_dim:int=768,
               num_heads:int=12,
               mlp_size:int=3072,
               mlp_dropout:int=0.1,
               attn_dropout:int=0):
    super().__init__()

    self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)
    
    self.mlp_block = MLPBlock(embedding_dim=embedding_dim, 
                              mlp_size=mlp_size,
                              dropout=mlp_dropout)
    
  def forward(self, x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x
    return x 

class ViT(nn.Module): 
  def __init__(self,
               img_size:int=224,
               in_channels:int=3,
               patch_size:int=16, 
               num_transformer_layers:int=12,
               embedding_dim:int=768,
               mlp_size:int=3072,
               num_heads:int=12,
               attn_dropout:int=0,
               mlp_dropout:int=0.1,
               embedding_dropout:int=0.1,
               num_classes:int=1000):
    super().__init__()
    assert img_size % patch_size == 0,  f"Image size must be divisible by patch size, image: {img_size}, patch size: {patch_size}"
    self.num_patches = (img_size * img_size) // patch_size**2
    self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)
    self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim))
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)
    self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                       num_heads=num_heads,
                                                                       mlp_size=mlp_size,
                                                                       mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
    self.classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )
  
  def forward(self, x):
    batch_size = x.shape[0]
    class_token = self.class_embedding.expand(batch_size, -1, -1)
    x = self.patch_embedding(x)
    x = torch.cat((class_token, x), dim=1)
    x = self.position_embedding + x
    x = self.embedding_dropout(x)
    x = self.transformer_encoder(x)
    x = self.classifier(x[:, 0])
    return x

def check():
    try:
        import torch
        import torchvision
        assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
        assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
        print(f"torch version: {torch.__version__}")
        print(f"torchvision version: {torchvision.__version__}")
    except:
        print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
        subprocess.check_call([
            "pip3", "install", "-U", "--pre",
            "torch", "torchvision", "torchaudio",
            "--extra-index-url",
            "https://download.pytorch.org/whl/nightly/cu113" ])
        import torch
        import torchvision
        print(f"torch version: {torch.__version__}")
        print(f"torchvision version: {torchvision.__version__}")

def intro():
    print("")
    my_utils.pretty_print( """
This implements parts 3-11 of the notebook 08_pytorch_paper_replicating.ipynb 
from pytorch-deep-learning. It replicates the results of the paper 
Improving Language Models by Padding Tokens with Pretrained Encoders by He et al. 2019. and
An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale
by Dosovitskiy et al., 2020.
    """ )
    my_utils.wait_for_user_input()

def main():
    intro()
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    print(f"Image path: {image_path}")
    manual_transforms = transforms.Compose([ transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                             transforms.ToTensor() ])

    print(f"Manually created transforms: {manual_transforms}")

    train_dataloader, test_dataloader, class_names = my_utils.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=BATCH_SIZE )

    print(f"Train dataloader length: {len(train_dataloader)}; test dataloader length: {len(test_dataloader)} class names: {class_names}")
    image_batch, label_batch = next(iter(train_dataloader))
    image, label = image_batch[0], label_batch[0]
    print(f"Image shape: {image.shape}; label: {label}")
    plt.imshow(image.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis(False)
    height = 224
    width = 224
    color_channels = 3
    patch_size = 16 
    number_of_patches = int((height * width) / patch_size ** 2)
    print(f"Number of patches: {number_of_patches}")
    embedding_layer_input_shape = (height, width, color_channels)
    embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)
    print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
    print(f"Output shape (single 1D sequence of patches): {embedding_layer_output_shape} -> (number_of_patches, embedding_dimension)")
    plt.imshow(image.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis(False)
    print("Pass the image through the convolutional layer")
    patch_size = 16
    conv2d = nn.Conv2d(in_channels=3, 
                    out_channels=768, 
                    kernel_size=patch_size,
                    stride=patch_size,
                    padding=0)
    
    image_out_of_conv = conv2d(image.unsqueeze(0)) 
    print(f"Image out of convolutional layer shape: {image_out_of_conv.shape}")
    print("requires_grad(): Should PyTorch track operations on this tensor so it can compute gradients for ")
    print("it during backpropagation? https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py")
    print(f"Does the convulutional layer require grad - image_out_of_conv.requires_grad? {image_out_of_conv.requires_grad}")
    random_indexes = random.sample(range(0, 758), k=5)
    my_utils.wait_for_user_input(f"Showing random convolutional feature maps from indexes: {random_indexes}")

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
    for i, idx in enumerate(random_indexes):
        image_conv_feature_map = image_out_of_conv[:, idx, :, :] 
        axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy()) 
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()
    flatten_layer = nn.Flatten(start_dim=2, end_dim=3)
    set_seeds()

    patchify = PatchEmbedding(in_channels=3,
                            patch_size=16,
                            embedding_dim=768)
    print("")
    print(f"Input image size: {image.unsqueeze(0).shape}")
    patch_embedded_image = patchify(image.unsqueeze(0)) 
    print(f"Output patch embedding sequence shape: {patch_embedded_image.shape}")
    rand_image_tensor = torch.randn(1, 3, 224, 224)
    rand_image_tensor_bad = torch.randn(1, 3, 250, 250)
    batch_size = patch_embedded_image.shape[0]
    embedding_dimension = patch_embedded_image.shape[-1]
    print(f"Batch size: {batch_size}; embedding dimension: {embedding_dimension}")
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                               requires_grad=True)
    print(f"Class token shape: {class_token.shape}")
    print( "torch.cat((class_token, patch_embedded_image), dim=1)" )
    patch_embedded_image_with_class_embedding = torch.cat((class_token, patch_embedded_image), dim=1) 

    print(patch_embedded_image_with_class_embedding)
    print( "Sequence of patch embeddings with class token prepended shape: " )
    print(f"{patch_embedded_image_with_class_embedding.shape} -> (batch_size, class_token + number_of_patches, embedding_dim)")
    number_of_patches = int((height * width) / patch_size**2)

    embedding_dimension = patch_embedded_image_with_class_embedding.shape[-1]
    position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension), requires_grad=True)
    print( "position_embedding, position_embedding.shape" )
    print( position_embedding, position_embedding.shape )
    print( "patch_embedded_image_with_class_embedding, patch_embedded_image_with_class_embedding.shape" )
    print( patch_embedded_image_with_class_embedding, patch_embedded_image_with_class_embedding.shape )
    print( "patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding" )
    patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding
    print(patch_and_position_embedding)
    print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")

    my_utils.wait_for_user_input("Putting it all together: from image to embedding")
    set_seeds()
    patch_size = 16
    print(f"Image tensor shape: {image.shape}")
    height, width = image.shape[1], image.shape[2]
    x = image.unsqueeze(0)
    print(f"Input image shape: {x.shape}")
    patch_embedding_layer = PatchEmbedding(in_channels=3,
                                        patch_size=patch_size, 
                                        embedding_dim=768)

    patch_embedding = patch_embedding_layer(x)
    print(f"Patch embedding shape: {patch_embedding.shape}")
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1]
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension), requires_grad=True)
    print(f"Class token embedding shape: {class_token.shape}")
    patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
    print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")
    number_of_patches = int((height*width) / patch_size**2)
    position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension), requires_grad=True)
    patch_and_position_embedding = patch_embedding_class_token + position_embedding
    print(f"Patch and position embedding shape: {patch_and_position_embedding.shape} ")

    my_utils.wait_for_user_input("Creating a Multi-Head Self-Attention (MSA) block")
    multihead_self_attention_block = MultiHeadSelfAttentionBlock(embedding_dim=768, num_heads=12, attn_dropout=0)
    patched_image_through_msa_block = multihead_self_attention_block(patch_and_position_embedding)
    print(f"Input shape of MSA block: {patch_and_position_embedding.shape}")
    print(f"Output shape of MSA block: {patched_image_through_msa_block.shape}")

    print("")
    my_utils.pretty_print("""Creating a Multi-Layer Perceptron (MLP) block,and passing the 
output of the MSA block through the MLP block to create the final output""")
    print("mlp_block = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.1)")
    print("patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block" )
    my_utils.wait_for_user_input()

    mlp_block = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.1)
    patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
    print(f"Input shape of MLP block: {patched_image_through_msa_block.shape}")
    print(f"Output shape of MLP block: {patched_image_through_mlp_block.shape}")

    my_utils.wait_for_user_input("Creating the Transformer Encoder. class TransformerEncoderBlock")
    print("""
Encoder = turn a sequence into learnable representation
Decoder = go from learn representation back to some sort of sequence
Residual connections = add a layer(s) input to its subsequent output, this enables the creation of 
deeper networks (prevents weights from getting too small): This is a summary of the model architecture:
    """)
    transformer_encoder_block = TransformerEncoderBlock()
    summary(
        model=transformer_encoder_block,
        input_size=(1, 197, 768),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

    my_utils.wait_for_user_input("torch_transformer_encoder_layer - nn.TransformerEncoderLayer")
    torch_transformer_encoder_layer = nn.TransformerEncoderLayer(
        d_model=768,  nhead=12,  dim_feedforward=3072, dropout=0.1, 
        activation="gelu", batch_first=True, norm_first=True)
    summary(
        model=torch_transformer_encoder_layer,
        input_size=(1, 197, 768),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

    my_utils.pretty_print("""
Putting it all together to create ViT    
    """)
    batch_size=32
    embedding_dim=768
    class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                requires_grad=True)
    class_embedding_expanded = class_embedding.expand(batch_size, -1, -1)
    print(class_embedding.shape)
    print(class_embedding_expanded.shape)
    set_seeds()
    random_image_tensor = torch.randn(1, 3, 224, 224)
    vit = ViT(num_classes=len(class_names))
    vit(random_image_tensor)
    summary(
        model=ViT(num_classes=len(class_names)),
        input_size=(1, 3, 224, 224), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
    num_params = 85800963
    print(f"Number of parameters: {num_params}")

    my_utils.wait_for_user_input("Training the model - optimizer torch.optim.Adam, loss_fn torch.nn.CrossEntropyLoss")
    print("engine.train(vit, train_dataloader, test_dataloader, epochs=10, optimizer, loss_fn, device)")
    device = my_utils.get_device()
    set_seeds()
    optimizer = torch.optim.Adam(vit.parameters(), 
                                 lr=1e-3,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    results = engine.train(
                        model=vit,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        epochs=10,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        device=device)

    print("Plot loss curves")
    my_utils.plot_loss_curves(results) 
    print("")

    my_utils.wait_for_user_input("Get a pretrained ViT model torchvision.models.vit_b_16")
    cost = 30 * 24 * 8
    print(f"Cost of renting a TPUv3 for 30 straight days: ${cost}USD")
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
    pretrained_vit = torchvision.models.vit_b_16(weights=None)
    pretrained_vit.load_state_dict(torch.load("./models/vit_b_16-c867db91.pth", weights_only=True))
    pretrained_vit = pretrained_vit.to(device)
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    set_seeds()
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    print("Summary of pretrained ViT model")
    summary(
        model=pretrained_vit,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

    vit_transforms = pretrained_vit_weights.transforms()
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = my_utils.create_dataloaders(train_dir=train_dir,
                                                                                                       test_dir=test_dir,
                                                                                                       transform=vit_transforms,
                                                                                                       batch_size=32) 
    print("Optimizer from pretrained_vit.parameters, loss_fn torch.nn.CrossEntropyLoss")                                                                                                   
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    set_seeds() 
    pretrained_vit_results = engine.train( model=pretrained_vit,
                                           train_dataloader=train_dataloader_pretrained,
                                           test_dataloader=test_dataloader_pretrained,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           epochs=10,
                                           device=device )

    print("Plot loss curves of pretrained ViT")
    my_utils.plot_loss_curves(pretrained_vit_results)        

    target_dir = "models"
    model_name = "08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth"
    my_utils.pretty_print(f"Save the model in file {model_name}")
    utils.save_model( model=pretrained_vit, target_dir=target_dir, model_name=model_name)
    pretrained_vit_model_size = Path( target_dir + "/" + model_name ).stat().st_size // (1024 * 1024)
    print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")

    my_utils.wait_for_user_input("Predicting on a custom image")
    custom_image_path = image_path / "04-pizza-dad.jpeg"

    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")
    my_utils.pred_and_plot_image(model=pretrained_vit,
                        image_path=custom_image_path,
                        class_names=class_names)

if __name__ == "__main__":
    main()  
