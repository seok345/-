# -

###  Polyglot-KO 챗봇 기능
기능 소개
ReadMate 프로젝트의 확장 기능으로, HuggingFace의 EleutherAI/polyglot-ko-1.3b 모델을 활용한 한국어 전용 CLI 기반 챗봇을 개발하였습니다.

해당 챗봇은 책 추천 외에도, 사용자가 자유롭게 입력한 질문에 대해 정확하고 안정적인 한국어 응답을 제공하는 것을 목표로 합니다.

## 개발 목적
사용자와의 자연어 상호작용을 가능하게 하여 질문과 대화를 이어나감

오픈소스 한국어 언어모델 실험 및 대화형 응용 확장

## 사용 모델 및 환경
모델: EleutherAI/polyglot-ko-1.3b

라이브러리: transformers, torch

실행 환경: Python 3.x, GPU(CUDA) 지원 시 자동 사용

## 챗봇 핵심 코드 설명

# 모델 및 토크나이저 로드
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 디바이스 설정 (GPU 우선 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(" Polyglot-KO 챗봇 시작! 종료하려면 '종료' 또는 'exit'를 입력하세요.\n")

while True:
    user_input = input(" 사용자: ").strip()
    if user_input.lower() in ["종료", "exit", "quit"]:
        print(" 챗봇을 종료합니다.")
        break

    # 프롬프트 구조 (Q/A 형식으로 모델 맥락 유도)
    prompt = f"""다음은 한국어 인공지능 챗봇과 사용자의 대화입니다.
Q: {user_input}
A:"""

    # 입력 변환 및 GPU 처리
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    # 응답 생성 (정확성 위주 설정)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,  # 예측 가능한 고정 응답
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 결과 디코딩 및 출력
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = full_response.replace(prompt, "").strip()
    print(f" 챗봇: {result}\n")

## 실행 방법

pip install transformers torch
python polyglot_ko_chat.py
polyglot_ko_chat.py 파일로 저장해서 실행

## 실행 화면 예시
 Polyglot-KO 챗봇 시작! 종료하려면 '종료' 또는 'exit'를 입력하세요.

 사용자: 인공지능이란?
 챗봇: 인공지능은 인간의 지능을 기계로 구현한 것입니다. 인공지능은 인간의 지능을 기계로 구현한 것입니다. 인공지능은 인간의 지능을 기계로 구현한 것입니다...

## 특징 및 유의사항
**정확도 중시 설정 (do_sample=False)**로 동일 입력 시 일관된 답변 제공

짧은 입력이나 비정형 문장에는 비논리적인 결과가 출력될 수 있음

지식형 대화형 기능으로 활용 가능
 
