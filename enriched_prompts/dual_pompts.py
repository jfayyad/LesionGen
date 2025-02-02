from openai import OpenAI
import base64
import os
from PIL import Image
import pandas as pd
import io

client = OpenAI(api_key='')

try:
    from transformers import T5Tokenizer, CLIPTokenizer
    # Initialize the dual tokenizers
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
except ImportError:
    clip_tokenizer = None
    t5_tokenizer = None


def truncate_clip_tokenizer(prompt_clip, clip_tokenizer, max_length=75, test_tokenlen=False):
    # Tokenize the prompt
    clip_tokenized = clip_tokenizer(prompt_clip, return_tensors="pt")

    # Check the length of the tokenized input
    input_length = len(clip_tokenized["input_ids"][0])

    if input_length < max_length:
        padding_length =( max_length - input_length)//2
        prompt_clip = prompt_clip + " [PAD]" * padding_length
    else:
        clip_tokenized = clip_tokenizer(prompt_clip, return_tensors="pt", truncation=True, max_length=max_length)
        prompt_clip = clip_tokenizer.decode(clip_tokenized["input_ids"][0], skip_special_tokens=True)

    # clip_tokenized = clip_tokenizer(prompt_clip, return_tensors="pt")
    if test_tokenlen:
        return prompt_clip, len(clip_tokenized["input_ids"][0])
    else:
        return prompt_clip, None


def generate_description_clip(t5_prompt, image_name,):

    prompt = """Convert the below response (including the category names) \
to a list of words or phrases for a image generation prompt, \
separated by comma when appropriate. Maximum number of words is 60. Output the list only."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in dermatology."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": t5_prompt}
                ]}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating description for {image_name}: {e}")
        return "Error generating description"


def generate_description_t5(diagnosis, image_name, encoded_image, max_tokens=500):
    """
    Generate a visual description of a skin lesion image using the T5 model.

    Args:
        diagnosis (tuple): A tuple containing the lesion type, age, gender, and localization.
        image_name (str): The name of the image file.
        encoded_image (str): The base64 encoded image.
        max_tokens (int): The maximum number of tokens to generate.
    
    Returns:
        str: The generated description.
    """

    lesion_type, age, gender, localization = diagnosis
    prompt = f"""\
You are a medical expert specializing in dermatology. \
I am providing you with an image of a skin lesion from the HAM10000 dataset. \
You are provided the information that this lesion is a {lesion_type}, \
and this comes from a {age} year old {gender} patient on their {localization}. \
Now perform the visual description again with step-by-step reasoning with the information. \
Use a schema with 11 entries to output your description: \
patient info (age, gender, lesion type, localization), lesion color, lesion color variability (with description), \
shape and shape variability (with description), size % with respect to the image, \
border definition, texture, specific dermoscopic patterns, elevation, \
the Fitzpatrick scale of the healthy skin tone around the lesion, \
additional notable features. Write in clear medical language and do not include additional information. \
Limit your response under 350 words.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in dermatology."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating description for {image_name}: {e}")
        return "Error generating description"


def process_images_from_HAM_csv_t5(df, image_dir, output_dir = None, test_mode=True, enable_clip_desc=True):
    """
    Process images from the HAM10000 dataset and generate descriptions for T5.

    Args:
    df (pd.DataFrame): The dataframe of the original HAM10000 metadata.
    image_dir (str): The folder containing the images.
    test_mode (bool): Whether to run in test mode (default: True).
    """

    data = []

    lesion_type_dict = {
        'akiec': "(AKIEC) actinic keratoses and intraepithelial carcinoma / Bowen's disease ",
        'bcc' : "(BCC) basal cell carcinoma",
        'bkl' : "(BKL) benign keratosis-like lesions, solar lentigines / seborrheic keratoses and lichen-planus like keratoses",
        'df' : "(DF) dermatofibroma",
        'mel' : "(MEL)melanoma",
        'nv' : "(NV) melanocytic nevi",
        'vasc' : "(VASC) vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage"
    }

    for idx, row in df.iterrows():
        image_name = row["image_id"]

        image_path = os.path.join(image_dir, (row['image_id']+'.jpg'))

        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        diagnosis = (
            lesion_type_dict[row['dx']],
            int(row['age']),
            row['sex'],
            row['localization']
        )

        t5_desc = generate_description_t5(diagnosis, image_name, encoded_image, max_tokens=480)

        if enable_clip_desc:
            # Generate the short 77 token CLIP description along with T5 description
            clip_desc = generate_description_clip(t5_desc, image_name)
            clip_desc, _ = truncate_clip_tokenizer(clip_desc, clip_tokenizer, max_length=77)

            data.append([image_name, t5_desc, clip_desc])
        else:
            data.append([image_name, t5_desc])

        if test_mode and idx >= 2:  
            break

    if enable_clip_desc:
        df = pd.DataFrame(data, columns=["Image Name", "T5 Description", "CLIP Description"])
    else:
        df = pd.DataFrame(data, columns=["Image Name", "Description"])

    # Save
    if not output_dir:
        output_dir = image_dir
    output_path = os.path.join(output_dir, "image_descriptions.csv")
    df.to_csv(output_path, index=False)
    print(f"Descriptions saved to {output_path}")


def process_images_clip_only(df, output_dir, test_mode=True):
    """
    Process images from a csv containing image_id and T5 description and generate descriptions for CLIP only.

    Args:
    df (pd.DataFrame): The dataframe of the image_id and T5.
    test_mode (bool): Whether to run in test mode (default: True).
    """

    data = []

    for idx, row in df.iterrows():
        image_name = row["Image Name"]
        t5_desc = row["Description"]
    
        # Generate the short 77 token CLIP description along with T5 description
        clip_desc = generate_description_clip(t5_desc, image_name)
        clip_desc, _ = truncate_clip_tokenizer(clip_desc, clip_tokenizer, max_length=77)

        data.append([image_name, t5_desc, clip_desc])

        if test_mode and idx >= 2:  
            break

    df = pd.DataFrame(data, columns=["Image Name", "CLIP Description"])

    # Save
    output_path = os.path.join(output_dir, "image_descriptions_clip.csv")
    df.to_csv(output_path, index=False)
    print(f"CLIP Descriptions saved to {output_path}")


# Run in test mode to process only one image
# process_images(image_folder='../zipped_classes/vasc', diagnosis="Vascular lesions", test_mode=False)

df_subset = pd.read_csv('./HAM10000_metadata_subset.csv')
process_images_from_HAM_csv_t5(df_subset, './data/training_dataset/',
                            #    output_dir='./', 
                               test_mode=False, 
                               enable_clip_desc=True)

# alternatively run the following to process images with T5 descriptions only
# df_subset_clip = pd.read_csv('./image_descriptions.csv')
# process_images_clip_only(df_subset_clip, './', test_mode=False)