import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil

from caption_translator import generate_caption, translate_single

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_PATH = "uploaded.jpg"


@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(...),
    lang: str = Form(...)
):
    try:
        # Save uploaded image
        with open(UPLOAD_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate caption
        caption = generate_caption(UPLOAD_PATH)

        if not caption:
            caption = "Caption generation failed"

        print("Generated Caption:", caption)

        # Translate
        translation_dict = translate_single(caption, lang)

        translated_text = translation_dict.get(
            lang, "Translation failed"
        )

        return {
            "caption": caption,
            "translations": translated_text
        }

    except Exception as e:
        print("ERROR in /upload-image/:", e)
        return {
            "caption": "Error generating caption",
            "translations": "Error translating caption"
        }


@app.get("/")
def root():
    return {"message": "Backend is running!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
