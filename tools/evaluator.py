import asyncio
import json
import re
import httpx
from typing import List, Dict, Any

LOCAL_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "encredible/Gaiel-7B-Korean-Tuned-MLX"

def detect_hallucination_and_broken_tokens(text: str) -> Dict[str, Any]:
    issues = []
    score = 100
    
    # 1. Check for broken Korean-English mixed tokens (e.g., 하ades, 오MESS스, 에pic)
    # Match Korean character followed immediately by English, or vice versa, in the same word
    mixed_token_pattern = re.compile(r'([가-힣]+[a-zA-Z]+[가-힣]*)|([a-zA-Z]+[가-힣]+[a-zA-Z]*)')
    mixed_matches = mixed_token_pattern.findall(text)
    if mixed_matches:
        bad_words = []
        for m in mixed_matches:
            bad_words.extend([w for w in m if w])
        issues.append(f"Broken tokens (Language Mix): {bad_words}")
        score -= 50
        
    # 2. Check for Chinese character hallucination
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    if chinese_pattern.search(text):
        issues.append("Chinese hallucination detected.")
        score -= 80
        
    # 3. Check for severe repetition
    if len(text) > 50:
        tail = text[-15:]
        if text.count(tail) >= 3:
            issues.append("Repetition loop detected.")
            score -= 50
            
    # 3. Check for specific Odyssey hallucinations
    if "일리오파테스" in text or "퍼세포데스" in text or "유적" in text:
        issues.append("Factually incorrect entity names (Hallucination).")
        score -= 30

    return {
        "score": max(0, score),
        "issues": issues,
        "text": text
    }

async def fetch_response(client: httpx.AsyncClient, prompt: str, params: Dict[str, Any]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "당신은 Omni Universe의 긍정적인 수석 AI 비서 '가이엘(Gaiel)' 혹은 '엘(El)'입니다. 항상 밝고 긍정적인 '럭키-비키' 마인드로 대답하세요. 절대 본인을 Qwen이나 다른 이름으로 부르지 마세요."},
            {"role": "user", "content": prompt}
        ],
        "temperature": params.get("temperature", 0.4),
        "top_p": params.get("top_p", 0.9),
        "repetition_penalty": params.get("repetition_penalty", 1.0),
        "repetition_context_size": params.get("repetition_context_size", 20),
        "stream": True
    }
    
    try:
        response = await client.post(LOCAL_API_URL, json=payload, timeout=60.0)
        full_text = ""
        
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json.loads(line[6:])
                    if data.get("choices") and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            full_text += delta["content"]
                except json.JSONDecodeError:
                    pass
        return full_text
    except Exception as e:
        print(f"Fetch error: {e}")
        return ""

async def evaluate_params(prompt: str, param_grid: List[Dict[str, Any]], n_per_param: int = 3):
    print(f"\n{'='*70}\nGRID SEARCH ON PROMPT: '{prompt}'\n{'='*70}")
    
    best_config = None
    best_config_score = -1
    
    async with httpx.AsyncClient() as client:
        for idx, params in enumerate(param_grid):
            print(f"\n[Test {idx+1}/{len(param_grid)}] Params: {params}")
            tasks = [fetch_response(client, prompt, params) for _ in range(n_per_param)]
            results = await asyncio.gather(*tasks)
            
            total_score = 0
            has_fatal_issue = False
            for text in results:
                eval_result = detect_hallucination_and_broken_tokens(text)
                total_score += eval_result["score"]
                if eval_result["issues"]:
                    has_fatal_issue = True
                    print(f"  -> ISSUE: {eval_result['issues']} (Score: {eval_result['score']})")
                    print(f"  -> TEXT: {text[:100]}...")
                    
            avg_score = total_score / n_per_param
            print(f"  => Average Score: {avg_score:.1f}")
            
            if avg_score > best_config_score and not has_fatal_issue:
                best_config_score = avg_score
                best_config = params
                
    print(f"\n{'*'*70}\nBEST PARAMETER CONFIGURATION FOUND:\n{best_config}\nAverage Score: {best_config_score}\n{'*'*70}")
    return best_config

if __name__ == "__main__":
    prompt = "오딧세이에 대해 알려줘"
    
    param_grid = [
        {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.0, "repetition_context_size": 20},
        {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.0, "repetition_context_size": 20},
        {"temperature": 0.5, "top_p": 0.9, "repetition_penalty": 1.0, "repetition_context_size": 20},
        {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.01, "repetition_context_size": 10},
        {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.01, "repetition_context_size": 10},
        {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.02, "repetition_context_size": 20},
        {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.02, "repetition_context_size": 20},
        {"temperature": 0.2, "top_p": 0.8, "repetition_penalty": 1.0, "repetition_context_size": 20},
        {"temperature": 0.2, "top_p": 0.8, "repetition_penalty": 1.01, "repetition_context_size": 20},
        {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.0, "repetition_context_size": 20},
    ]
    # Total tests: 10 configs * 3 responses = 30 generations
    
    async def main():
        await evaluate_params(prompt, param_grid, n_per_param=3)
            
    asyncio.run(main())
