# musubi wan_cache_text_encoder_outputs.py, computed in fp32 on CPU, saved in bf16 (the same
# varlen_t5_bfloat16 format as the GPU caches). This CPU has no native bf16, and emulated-bf16
# T5 drifted 3-7% from the GPU cache, while fp32 is within the GPU cache's own bf16 rounding (~2.3%).
import sys, torch
if not torch.cuda.is_available():         # t5.py evaluates torch.cuda.current_device() as a default arg at import
    torch.cuda.current_device = lambda: "cpu"
sys.path.insert(0, "/workspace/musubi-tuner/src")
from musubi_tuner.wan.configs import wan_t2v_14B
wan_t2v_14B.t2v_14B.t5_dtype = torch.float32
import musubi_tuner.wan_cache_text_encoder_outputs as M
orig = M.save_text_encoder_output_cache_wan
M.save_text_encoder_output_cache_wan = lambda item, embed: orig(item, embed.to(torch.bfloat16))
sys.argv = ["wan_cache_text_encoder_outputs.py", *sys.argv[1:]]
M.main()
