import torch

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

tensor = torch.rand(3,4)
print(tensor)



# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# 행렬 연산과 배열연산
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# single element tensor를 가져올 때는 item()사용
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

