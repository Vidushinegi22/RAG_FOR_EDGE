import csv
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def generate_questions_and_answers(context, num_pairs=3):
    """
    Generate question-answer pairs based on the given text context.
    Returns a list of QA pairs; falls back to naive parsing if needed.
    """
    def build_prompt():
        return f"""Based on the following context, generate {num_pairs} different question-answer pairs:

Context: {context}

Generate {num_pairs} new and different question-answer pairs related to this context. Make the questions diverse and explore different aspects of the context.

Return your response in valid JSON format like this:
{{
  "question_answer_pairs": [
    {{
      "question": "First generated question?",
      "answer": "Answer to the first question"
    }},
    {{
      "question": "Second generated question?",
      "answer": "Answer to the second question"
    }},
    {{
      "question": "Third generated question?",
      "answer": "Answer to the third question"
    }}
  ]
}}

Your response (in JSON format):"""

    # Prepare prompt
    prompt = build_prompt()
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    # Generate model output
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Try parsing valid JSON
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            if "question_answer_pairs" in data and isinstance(data["question_answer_pairs"], list):
                return data["question_answer_pairs"][:num_pairs]
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response_text}")

    # Fallback: naive question/answer parsing
    pairs = []
    current_question = None
    current_answer = ""

    for i, line in enumerate(response_text.split('\n')):
        line = line.strip()
        if not line:
            continue

        if (line.endswith('?') or line.startswith('Q') or
            any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '- '])):

            if current_question:
                pairs.append({
                    "question": current_question,
                    "answer": current_answer.strip()
                })
                current_answer = ""

            if line.startswith(('1.', '2.', '3.', '- ', 'Q:')):
                current_question = line.split('.', 1)[-1].strip() if '.' in line[:3] else line
                current_question = current_question.split(':', 1)[-1].strip() if ':' in current_question[:3] else current_question
                current_question = current_question.replace('- ', '', 1).strip()
            else:
                current_question = line

        elif current_question:
            if line.startswith(('A:', 'Answer:')):
                current_answer += line.split(':', 1)[-1].strip() + " "
            else:
                current_answer += line + " "

        if len(pairs) == num_pairs - 1 and current_question and i == len(response_text.split('\n')) - 1:
            pairs.append({
                "question": current_question,
                "answer": current_answer.strip()
            })

    return pairs[:num_pairs]

def process_csv_to_json(input_file, output_file, pairs_per_entry=3):
    """
    Read a CSV with 'page', 'text', and 'images' columns, treat each page as a chunk,
    include images with full paths (e.g., "/output_dataset/img.png"), generate Q&A pairs from text only,
    and save incrementally.
    """
    all_chunks = []
    processed_pages = set()

    # Load existing data from the output file, if it exists
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            all_chunks = existing_data.get("chunks", [])
            processed_pages = {chunk["page"] for chunk in all_chunks}
            print(f"Loaded {len(all_chunks)} existing chunks from {output_file}")
    except FileNotFoundError:
        print(f"No existing file found at {output_file}, starting fresh")
    except json.JSONDecodeError:
        print(f"Error reading {output_file}, starting with an empty list")
        all_chunks = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                page = int(row['page'])
            except (KeyError, ValueError):
                print(f"Skipping row {i}: Invalid or missing page number")
                continue

            context = row.get('text', '').strip()
            if len(context.split()) <= 10:
                print(f"Skipping page {page}: Text is 10 words or less.")
                continue

            if page in processed_pages:
                print(f"Skipping page {page}: Already processed.")
                continue

            # Parse images with error handling
            images = []
            if 'images' in row:
                try:
                    images = eval(row['images'])
                    if not isinstance(images, list):
                        images = []
                except Exception as e:
                    print(f"Error parsing images for page {page}: {e}")
                    images = []
            else:
                images = []

            # Prepend "/output_dataset/" to each image filename
            full_images = ["/output_dataset/" + img for img in images if isinstance(img, str)]

            print(f"Processing page {page}: {context[:50]}...")

            # Generate question-answer pairs based only on text
            new_pairs = generate_questions_and_answers(context, pairs_per_entry)

            # Add new chunk with full image paths
            chunk = {
                "page": page,
                "chunk_text": context,
                "images": full_images,
                "question_answer_pairs": new_pairs
            }
            all_chunks.append(chunk)

            # Save incrementally after each page
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"chunks": all_chunks}, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(all_chunks)} chunks after processing page {page}")

    print(f"Finished! Generated {len(all_chunks)} chunks and saved to {output_file}")

if __name__ == "__main__":
    input_file = r"C:\Users\RYI3KOR\Desktop\training\output_dataset\grand_vitara_dataset.csv"  # Replace with your CSV file path
    output_file = r"C:\Users\RYI3KOR\Desktop\training\output_dataset\output.json"  # Replace with your desired JSON output path
    process_csv_to_json(input_file, output_file, pairs_per_entry=3)