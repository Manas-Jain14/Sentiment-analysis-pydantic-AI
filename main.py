import os
import asyncio
import json
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from sklearn.metrics import confusion_matrix

# Load environment variables from .env file
load_dotenv()

# ── Pydantic Output Model ──
class SentimentResult(BaseModel):
    label: str

# ── Agent Definition ──
api_key = os.getenv("GOOGLE_API_KEY")
provider = GoogleProvider(api_key=api_key)
model_name = 'gemini-3.1-flash-lite'
model = GoogleModel(model_name, provider=provider)

# RESTORED: Detailed original prompt that achieved 80% accuracy
agent = Agent(
    model,
    output_type=SentimentResult,
    system_prompt="""<role>
You are a sentiment analysis agent for automotive customer conversations.
</role>

<task>
Analyze the sentiment of the Question and Response and return exactly one of these labels:
CONFIDENCE | CONFUSION | TRUST | INTEREST | DOUBT | URGENCY | FRUSTRATION | NONE
</task>

<definitions>
CONFIDENCE - user feels satisfied or agrees
CONFUSION - user feels unclear or lost
TRUST - user seeks validation of information
INTEREST - user is curious about product details
DOUBT - user feels skeptical about information
URGENCY - user feels ready to act immediately
FRUSTRATION - user feels irritated or dissatisfied
NONE - no clear sentiment detected
</definitions>

<constraints>
- Return only ONE label
- No explanation
- No extra text
- Capital letters only
</constraints>

<examples>
Question: What is the mileage?
Response: Can you tell me the mileage?
Label: INTEREST

Question: Is this price correct?
Response: Can I trust this information?
Label: TRUST

Question: You never answer properly.
Response: Please explain again.
Label: FRUSTRATION
</examples>""",
    model_settings={"temperature": 0}
)

# 1. Read ground truth data
def read_ground_truth():
    df = pd.read_csv("ground_truth.csv", sep='\t', encoding='utf-8')
    if 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    return df

# 2. Predict sentiment for a single record with Manual Retry Logic
async def predict_sentiment(question, response):
    prompt = f"Question: {question}\nResponse: {response}\nClassify the intent."
    
    for attempt in range(3):
        try:
            result = await agent.run(prompt)
            # Normalize for comparison
            label = result.output.label.strip().upper()
            return {"label": label}
        
        except Exception as e:
            err_str = str(e).lower()
            
            if "429" in err_str:
                print(f"Rate limit (429). Waiting 20s... (Attempt {attempt+1}/3)")
                await asyncio.sleep(20)
            elif "503" in err_str: # FIXED: was err_msg before
                print(f"Server busy (503). Waiting 15s... (Attempt {attempt+1}/3)")
                await asyncio.sleep(15)
            else:
                print(f"API Error: {e}")
                break 
                
    return {"label": "NONE"}

# 3. Throttled Prediction Logic (5 RPM to stay under Free Tier limits)
async def generate_predictions(df):
    semaphore = asyncio.Semaphore(1)
    
    async def throttled_predict(index, question, response):
        async with semaphore:
            print(f"Processing record {index}/{len(df)}...")
            result = await predict_sentiment(question, response)
            # Strict 12-second delay for stability
            await asyncio.sleep(12)
            return result

    tasks = []
    for i, row in df.iterrows():
        tasks.append(throttled_predict(i + 1, row['question'], row['response']))
    
    results = await asyncio.gather(*tasks)
    return [res['label'] for res in results]

# 4. Generate metrics
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total

def generate_confusion_matrix(y_true, y_pred):
    labels = ["CONFIDENCE", "CONFUSION", "TRUST", "INTEREST", "DOUBT", "URGENCY", "FRUSTRATION", "NONE"]
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()

# 5. Save results
def save_results(total, correct, accuracy, cm, df):
    output = {
        "total_records": total,
        "correct_predictions": correct,
        "accuracy": f"{accuracy:.2f}%",
        "confusion_matrix": cm,
        "predictions": df.to_dict(orient='records')
    }
    with open("results.json", "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

# 6. Main
async def main():
    print("Reading ground truth data...")
    df = read_ground_truth()
    
    # Balanced sampling - 2 rows per label
    df = df.groupby('label').head(2).reset_index(drop=True)
    
    # Print label distribution to verify
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nTotal records: {len(df)}\n")
    
    print(f"Generating predictions for {len(df)} records...")
    
    predicted_labels = await generate_predictions(df)
    df['predicted_label'] = predicted_labels
    
    accuracy, correct, total = calculate_accuracy(df['label'], df['predicted_label'])
    cm = generate_confusion_matrix(df['label'], df['predicted_label'])
    
    # Save Heatmap
    labels = ["CONFIDENCE", "CONFUSION", "TRUST", "INTEREST", "DOUBT", "URGENCY", "FRUSTRATION", "NONE"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Sentiment Analysis Heatmap - Accuracy: {accuracy:.2f}%')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    save_results(total, correct, accuracy, cm, df)
    print(f"\nEvaluation Complete!")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
