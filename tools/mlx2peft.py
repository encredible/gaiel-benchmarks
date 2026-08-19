"""MLX LoRA 어댑터를 PEFT 형식으로 변환한다.

수식 대응:
  MLX  : y = Wx + scale * (x @ lora_a) @ lora_b        lora_a[in,r], lora_b[r,out]
  PEFT : y = Wx + (alpha/r) * (x @ A.T) @ B.T          A[r,in],      B[out,r]
따라서 A = lora_a.T, B = lora_b.T, alpha = scale * r.

MLX 는 일부 레이어에만 LoRA 를 걸 수 있으므로 PEFT 의 layers_to_transform 으로 동일하게 제한한다.
"""
import json, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

def convert(src_dir: Path, out_dir: Path):
    cfg = json.loads((src_dir / "adapter_config.json").read_text())
    base = cfg["model"]
    lp = cfg.get("lora_parameters") or {}
    rank = int(lp.get("rank", 8))
    scale = float(lp.get("scale", 20.0))
    dropout = float(lp.get("dropout", 0.0))
    alpha = scale * rank

    new, layers, mods = {}, set(), set()
    with safe_open(src_dir / "adapters.safetensors", framework="np") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            parts = k.split(".")                     # model.layers.N.<blk>.<proj>.lora_[ab]
            li, proj, suf = int(parts[2]), parts[-2], parts[-1]
            layers.add(li); mods.add(proj)
            stem = "base_model.model." + ".".join(parts[:-1])
            if suf == "lora_a":
                new[f"{stem}.lora_A.weight"] = np.ascontiguousarray(t.T)   # [r, in]
            elif suf == "lora_b":
                new[f"{stem}.lora_B.weight"] = np.ascontiguousarray(t.T)   # [out, r]
            else:
                raise ValueError(f"예상 못한 키: {k}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(new, str(out_dir / "adapter_model.safetensors"))
    peft = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": rank, "lora_alpha": alpha, "lora_dropout": dropout,
        "target_modules": sorted(mods),
        "layers_to_transform": sorted(layers),
        "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
        "modules_to_save": None, "init_lora_weights": True,
        "use_rslora": False, "use_dora": False,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(peft, indent=2))
    print(f"  베이스={base}")
    print(f"  r={rank} scale={scale} → lora_alpha={alpha}")
    print(f"  레이어 {min(layers)}~{max(layers)} ({len(layers)}개) | 모듈 {sorted(mods)}")
    print(f"  텐서 {len(new)}개 → {out_dir}")
    # 검증: ΔW 가 두 표현에서 동일한지 무작위 표본으로 확인
    with safe_open(src_dir / "adapters.safetensors", framework="np") as f:
        k = [x for x in f.keys() if x.endswith("lora_a")][0]
        a = f.get_tensor(k); b = f.get_tensor(k[:-1] + "b")
    stem = "base_model.model." + ".".join(k.split(".")[:-1])
    A, B = new[f"{stem}.lora_A.weight"], new[f"{stem}.lora_B.weight"]
    d_mlx = scale * (a @ b)                 # [in, out]
    d_peft = (alpha / rank) * (A.T @ B.T)   # [in, out]
    err = np.abs(d_mlx - d_peft).max()
    print(f"  ΔW 검증: 최대 절대오차 {err:.3e}  {'✅ 일치' if err < 1e-4 else '❌ 불일치'}")

if __name__ == "__main__":
    for tag, src in [("1.5B","g1.5B"), ("7B","g7B")]:
        print(f"\n### Gaiel-{tag}")
        convert(Path("/Users/K/omni-work/runpod/mlx_adapters")/src,
                Path("/Users/K/omni-work/runpod/peft_adapters")/f"gaiel-{tag.lower()}-korean")
