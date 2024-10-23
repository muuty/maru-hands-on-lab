from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os


openai_api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 생성
app = FastAPI()
handler = Mangum(app)

# Langchain 설정
llm_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, max_tokens=500)

# 프롬프트 설정
prompt = PromptTemplate.from_template(
    "당신은 애완동물 추천을 기가막히게 잘합니다! 사람들의 말속에서 키워드를 쏙쏙 뽑아내서 강아지와 고양이를 추천해주죠! "
    "말끝에 멍! 을 붙일거에요. 이전 대화: {history} 새로운 질문: {input}"
)

# 사용자별 메모리 저장 딕셔너리
user_memories = {}


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the FastAPI demo!"}

# 요청 모델 정의
class QuestionRequest(BaseModel):
    user_id: str
    input: str


# API 엔드포인트 정의
@app.post("/openai/chat")
async def chat_with_openai(request: QuestionRequest):
    try:
        # 사용자별로 메모리를 관리
        if request.user_id not in user_memories:
            user_memories[request.user_id] = ConversationBufferMemory(return_messages=True)

        # 해당 사용자의 메모리를 사용해 체인 생성
        user_memory = user_memories[request.user_id]
        chain = ConversationChain(llm=llm_model, memory=user_memory, prompt=prompt)

        # Langchain을 이용해 사용자 질문 처리
        result = chain.invoke({"input": request.input})
        return {"response": result['response']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
