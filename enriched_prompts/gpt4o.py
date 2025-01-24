from openai import OpenAI
import base64
import os
from PIL import Image
import pandas as pd
import io

client = OpenAI(api_key='')


def generate_description(diagnosis, image_name, encoded_image):
    prompt = f"Analyze the image file named '{image_name}' diagnosed as '{diagnosis}'. Describe unique visual features specific to this image, including size, precise shape (e.g., circular, irregular), border details, specific color patterns, any texture variations, and any observable elements like hair presence or unusual markings. Avoid general statements and instead focus on the specific details visible in this image."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating description for {image_name}: {e}")
        return "Error generating description"

def process_images(image_folder, diagnosis, test_mode=True):
    data = []

    for idx, image_name in enumerate(os.listdir(image_folder)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)

            try:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                print(f"Processing {image_name}...")
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
                continue

            # Generate description for the image
            description = generate_description(diagnosis, image_name, encoded_image)

            
            data.append([image_name, description])

            if test_mode and idx >= 2:  
                break


    df = pd.DataFrame(data, columns=["Image Name", "Description"])

    # Save
    output_path = os.path.join(image_folder, "image_descriptions.csv")
    df.to_csv(output_path, index=False)
    print(f"Descriptions saved to {output_path}")

# Run in test mode to process only one image
process_images(image_folder='../zipped_classes/vasc', diagnosis="Vascular lesions", test_mode=False)
