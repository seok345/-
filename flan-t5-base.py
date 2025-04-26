from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 및 토크나이저 로드
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 디바이스 설정 (GPU 사용 시 CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(" Polyglot-KO 챗봇 시작! 종료하려면 '종료' 또는 'exit'를 입력하세요.\n")

while True:
    user_input = input(" 사용자: ").strip()
    if user_input.lower() in ["종료", "exit", "quit"]:
        print(" 챗봇을 종료합니다.")
        break

    # 대화 프롬프트 구성
    prompt = f"""다음은 한국어 인공지능 챗봇과 사용자의 대화입니다.
Q: {user_input}
A:"""

    # 입력 토크나이즈 및 텐서 변환
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    # 텍스트 생성 (정확성 위주 설정)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,  # 샘플링 비활성화
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 디코딩 및 응답 출력
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = full_response.replace(prompt, "").strip()

    print(f" 챗봇: {result}\n")
