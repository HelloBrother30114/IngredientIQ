import json
import base64
import os
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq

app = FastAPI()

# Frontend se connect karne ke liye CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-product")
async def analyze_product(
    file: UploadFile = File(...), 
    x_api_key: str = Header(None, alias="X-API-Key") # Frontend se aane wali key yahan catch hogi
):
    # Agar key missing hai
    if not x_api_key:
        raise HTTPException(status_code=401, detail="⚠️ API Key missing! Please enter your Groq API Key.")

    try:
        # Har request par user ki current key se client initialize hoga
        client = Groq(api_key=x_api_key)
        
        # Image ko read karke base64 mein convert karna
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Proper data URI format
        image_url = f"data:{file.content_type};base64,{base64_image}"
        
        # Qwen-Specific Strict Prompt with ANTI-LOOP Instructions
        prompt = """
        You are an expert cosmetic and food product analyzer. 
        Analyze this product label image. Extract all the ingredients (OCR) and provide a strict JSON response.
        
        CRITICAL INSTRUCTIONS FOR QWEN:
        1. KEEP YOUR THINKING VERY BRIEF. Do not get stuck in loops repeating yourself.
        2. Do not over-analyze the formatting of the text. Just extract the core ingredients and move on.
        3. YOU MUST RETURN ONLY RAW JSON AT THE END.
        4. The JSON must start exactly with '{' and end exactly with '}'.
        
        The JSON must perfectly match this structure:
        {
          "product_name": "String (Guess the product name based on label/packaging, or write 'Unknown Product')",
          "product_score": Integer (0-100 health/safety score),
          "final_verdict": "String (1-2 sentences summarizing if it's safe or bad)",
          "avoid_if": ["String", "String"],
          "pros": ["String", "String"],
          "cons": ["String", "String"],
          "beneficial_ingredients": [{"ingredient": "Name", "effect": "What it does"}],
          "harmful_ingredients": [{"ingredient": "Name", "effect": "Why it's bad"}],
          "extracted_ocr": ["String", "String"]
        }
        """

        # Qwen Model Call
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=4096 
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # Smart JSON Extractor
        json_string = ""
        clean_content = raw_content
        
        if "</think>" in raw_content:
            clean_content = raw_content.split("</think>")[-1].strip()
        elif raw_content.startswith("<think>"):
            clean_content = ""
            
        start_idx = clean_content.find('{')
        end_idx = clean_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            json_string = clean_content[start_idx:end_idx+1]
            
        # Agar JSON nahi mila
        if not json_string:
            return {
                "extracted_ocr": ["AI got stuck reading this dense text. Please try cropping the image!"],
                "analysis": {
                    "product_name": "Scan Incomplete",
                    "product_score": 0,
                    "final_verdict": "The AI got confused by the dense or complex text on this label and couldn't finish the analysis. Try cropping the image to focus only on the ingredients list and scan again!",
                    "avoid_if": ["Scanning full shiny wrappers or blurry text"],
                    "pros": [],
                    "cons": ["Image text too complex for a single scan"],
                    "beneficial_ingredients": [],
                    "harmful_ingredients": []
                }
            }
            
        # Parse clean JSON
        analysis_data = json.loads(json_string)
        
        return {
            "extracted_ocr": analysis_data.get("extracted_ocr", []),
            "analysis": analysis_data
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        raise HTTPException(status_code=500, detail="AI response could not be parsed. Please try cropping again.")
    except Exception as e:
        error_str = str(e)
        print(f"Server Error: {error_str}")
        
        # API Error Handling (Invalid Key ya Rate Limit)
        if "401" in error_str or "unauthorized" in error_str.lower():
            raise HTTPException(status_code=401, detail="⚠️ Invalid API Key! Please check your Groq key.")
        if "429" in error_str or "rate_limit" in error_str.lower():
            raise HTTPException(
                status_code=429, 
                detail="⚠️ API ki limit khatam ho chuki hai! Thodi der baad try karein."
            )
            
        raise HTTPException(status_code=500, detail=f"Server Error: {error_str}")

# REACT DIST MOUNT (Static files ko root URL par serve karne ke liye)
if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")

# HF SPACE ENTRYPOINT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)