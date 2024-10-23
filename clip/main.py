import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm
from torch.nn import CosineSimilarity


DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)


def extract_image_features():
    # extract features from images and save them to a json file
    features = {}
    images_dir = os.path.join(os.getcwd(), "images")

    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        try:
          image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
        except:
          print(f'Error: {image_name}')
          continue

        with torch.no_grad():
            image_features = model.encode_image(image)
            features[image_name] = image_features.cpu().numpy().tolist()

    with open("pet_features.json", "w", encoding='utf-8') as fp:
        json.dump(features, fp)


def recommend_pet(user_preference: str) -> str:
    with open("pet_features.json", "r", encoding='utf-8') as fp:
        pet_features = json.load(fp)
        print(pet_features.keys())

    # transform user_preference to embedding vector
    preference_token = clip.tokenize(user_preference).to("cpu")
    preference_vector = model.encode_text(preference_token)

    similarities = {}
    for pet_name, pet_feature in pet_features.items():
        # distance as similarity
        similarity = torch.norm(torch.tensor(pet_feature).flatten() - torch.tensor(preference_vector).flatten(), p=2)

        # CosineSimilarity as similarity
        # similarity = CosineSimilarity(dim=0)(torch.tensor(pet_feature).flatten(), torch.tensor(preference_vector).flatten())
        similarities[pet_name] = similarity.item()

    similarities = sorted(similarities.items(), key=lambda x: x[1])
    return similarities[:5]

if __name__ == "__main__":
    # extract_image_features()

    print(recommend_pet("cute baby cat"))