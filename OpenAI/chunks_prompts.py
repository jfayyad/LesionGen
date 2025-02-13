import openai
import pandas as pd
import os
import time

# ✅ Load CSV with image metadata
csv_path = "/home/jfayyad/Python_Projects/LesionGen/HAM10000_metadata_subset.csv"
df = pd.read_csv(csv_path)

# ✅ Ensure required columns exist
required_columns = ["image_id", "dx", "age", "sex", "localization"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"CSV must have the following columns: {', '.join(required_columns)}")

# ✅ Define lesion type mapping
lesion_type_dict = {
    'akiec': "(AKIEC) actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    'bcc': "(BCC) basal cell carcinoma",
    'bkl': "(BKL) benign keratosis-like lesions, solar lentigines / seborrheic keratoses and lichen-planus like keratoses",
    'df': "(DF) dermatofibroma",
    'mel': "(MEL) melanoma",
    'nv': "(NV) melanocytic nevi",
    'vasc': "(VASC) vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage"
}

# ✅ Initialize OpenAI Client
client = openai.OpenAI(api_key="")

# ✅ Output file for results
output_csv = "/home/jfayyad/Python_Projects/LesionGen/HAM10000_t5_descriptions.csv"

# ✅ Load existing results if available (to resume from where it stopped)
if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    completed_ids = set(df_existing["image_id"])
else:
    completed_ids = set()

# ✅ Process in chunks of 1000
batch_size = 500
df_remaining = df[~df["image_id"].isin(completed_ids)]
num_batches = (len(df_remaining) // batch_size) + 1

print(f"🚀 Starting processing... Total remaining images: {len(df_remaining)}")

for batch_idx in range(num_batches):
    batch_df = df_remaining.iloc[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    # ✅ Store results for the current batch
    results = []

    for _, row in batch_df.iterrows():
        lesion_type = lesion_type_dict.get(row["dx"], row["dx"])  # Map lesion type

        # ✅ Format the prompt
        prompt_t5 = f"""
        You are provided with an image of a skin lesion from the HAM10000 dataset.
        The lesion has been identified as {lesion_type} from a {row['age']}-year-old {row['sex']} patient on their {row['localization']}.
        Please analyze the image and generate a structured visual description, 
        and perform step-by-step reasoning with the given information. 
        Use a schema with 12 entries to output your description: 
        Patient info (age, gender, lesion type, localization), Lesion color, Color variability detail,
        Lesion shape, Shape variability detail, Size % with respect to the image,
        Border definition, Lesion texture, Specific dermoscopic patterns, Lesion elevation,
        Fitzpatrick scale of the healthy skin tone around the lesion,
        and Additional notable features. Write in clear medical language and do not include additional information in your output.
        Limit your response under 350 words.
        """

        try:
            # ✅ Send request to OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical expert specializing in dermatology."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_t5}
                    ]}
                ],
                max_tokens=500,
                temperature=1.0
            )

            # ✅ Extract the response content
            description = response.choices[0].message.content.strip()

            # ✅ Store results
            results.append({"image_id": row["image_id"], "description": description})

        except Exception as e:
            print(f"❌ Error processing {row['image_id']}: {e}")
            time.sleep(5)  # Sleep to prevent API rate limit issues

    # ✅ Save batch results
    if results:
        results_df = pd.DataFrame(results)
        if os.path.exists(output_csv):
            results_df.to_csv(output_csv, mode="a", header=False, index=False)  # Append to existing file
        else:
            results_df.to_csv(output_csv, index=False)  # Create new file

    print(f"✅ Batch {batch_idx + 1}/{num_batches} completed.")

print(f"🎉 Processing complete! Descriptions saved to {output_csv}")
