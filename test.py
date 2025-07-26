import torch

# 模拟你的模型生成的一些 token ID
# 假设它们被错误地存储为 bfloat16
# 比如 [BOS, "Hello", "world", EOS]
# token id: [1, 15043, 3186, 2]
generated_ids_wrong_dtype = torch.tensor([1, 15043, 3186, 2], dtype=torch.bfloat16)
print(f"错误的张量 (dtype={generated_ids_wrong_dtype.dtype}):")
print(generated_ids_wrong_dtype)
# 输出: tensor([1.0000, 15040.0000, 3186.0000, 2.0000], dtype=torch.bfloat16)
# 注意：bfloat16 精度有限，15043 变成了 15040
# 现在，我们模拟 tokenizer 的错误行为：把这块内存当作 long 来读取
# .view() 可以实现这个“二进制重解释”
# 注意：一个 long 是 64 位，一个 bfloat16 是 16 位，所以我们需要4个bfloat16才能凑成一个long
# 我们只看第一个元素会发生什么，假设内存是连续的
generated_ids_interpreted_as_long = generated_ids_wrong_dtype.view(torch.int64)
print("\n当把它的二进制数据当作 int64 (long) 来解读时:")
print(generated_ids_interpreted_as_long)
