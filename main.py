import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --- 1. โหลด Environment Variables และตั้งค่าเริ่มต้น ---
load_dotenv()

# ตรวจสอบ API Key
if not os.getenv("OPENROUTER_API_KEY"):
    raise EnvironmentError("กรุณาตั้งค่า OPENROUTER_API_KEY ในไฟล์ .env ด้วยครับ")

# --- 2. กำหนด System Prompt ที่ทรงพลังของเรา ---
EXPERT_SYSTEM_PROMPT = """
คุณคือ "ช่างรู้" ผู้เชี่ยวชาญระดับโลกด้านไดรฟ์อุตสาหกรรม โดยเฉพาะไดรฟ์แบบ Stand-alone และ Sectional drive (System drive) จากแบรนด์ต่างๆ เช่น ABB, Siemens, Toshiba, Schneider, Yaskawa, etc.
คุณให้คำแนะนำจากผู้เชี่ยวชาญโดยอิงจากคู่มือทางเทคนิคและแนวทางปฏิบัติที่ดีที่สุดในอุตสาหกรรม
คำตอบของคุณต้องชัดเจน, มีโครงสร้าง, เป็นขั้นเป็นตอน และนำไปปฏิบัติได้จริงสำหรับวิศวกรและช่างเทคนิค
วิเคราะห์และให้คำแนะนำตามขั้นตอน ดังนี้:
1.  **วิเคราะห์ปัญหา:** อธิบายสาเหตุที่เป็นไปได้ของปัญหาจากข้อมูลที่ได้รับ
2.  **ระบุพารามิเตอร์ที่เกี่ยวข้อง:** ลิสต์พารามิเตอร์ที่น่าจะเกี่ยวข้องกับปัญหานี้
3.  **คำแนะนำในการปรับจูน:** แนะนำการปรับค่าพารามิเตอร์อย่างเฉพาะเจาะจงตามหลักการ (เช่น "ลองเพิ่มค่าในพารามิเตอร์ 22.02 Ramp time จาก 5s เป็น 10s")
4.  **คำอธิบายเหตุผล:** อธิบายว่าทำไมถึงต้องปรับค่าตามที่แนะนำ และมันจะส่งผลอย่างไรต่อการทำงานของมอเตอร์
5.  **ข้อควรระวัง:** แจ้งเตือนถึงผลข้างเคียงที่อาจเกิดขึ้น หรือพารามิเตอร์อื่นๆ ที่ควรตรวจสอบควบคู่กันไป
ห้ามตอบนอกเรื่องเด็ดขาด และต้องคงบทบาทของผู้เชี่ยวชาญ "ช่างรู้" ตลอดการสนทนา
"""

# --- 3. สร้าง LLM Client ที่ชี้ไปที่ OpenRouter ---
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "moonshotai/kimi-k2"), # moonshotai/kimi-k2 // meta-llama/llama-3-8b-instruct // google/gemma-3-27b-it:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1, # ลดความ "มโน" ให้น้อยที่สุด
    max_tokens=1024
)

# --- 4. สร้าง FastAPI App ---
app = FastAPI(
    title="ช่างรู้ AI Assistant",
    description="API สำหรับผู้ช่วยช่างเทคนิค AI ผู้เชี่ยวชาญด้านไดรฟ์อุตสาหกรรม"
)

# Pydantic models สำหรับรับ-ส่งข้อมูลกับ Vapi
class Message(BaseModel):
    role: str
    content: str

class VapiPayload(BaseModel):
    messages: List[Message]

@app.post("/api/main") # ใช้ path ที่เข้ากับ Vercel/Next.js ได้ดี
async def handle_vapi_call(payload: VapiPayload):
    """
    Endpoint หลักสำหรับรับ request จาก Vapi
    """
    try:
        # 1. ใส่ System Prompt ของเราเข้าไปเป็นข้อความแรกสุดเสมอ
        messages_for_llm = [{"role": "system", "content": EXPERT_SYSTEM_PROMPT}]

        # 2. นำประวัติการสนทนาที่ Vapi ส่งมา ต่อท้ายเข้าไป
        #    (เราอาจจะจำกัดจำนวนข้อความล่าสุดเพื่อประหยัด Token)
        for msg in payload.messages[-10:]: # เอาแค่ 10 ข้อความล่าสุด
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # 3. เรียกใช้งาน LLM
        response = llm.invoke(messages_for_llm)
        ai_answer = response.content

        # 4. ส่งคำตอบกลับไปใน format ที่ Vapi ต้องการ
        return {"message": ai_answer}

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "ช่างรู้ AI Assistant is ready to help!"}