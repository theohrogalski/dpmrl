def recurse_uncertainty_calculation(num_step:int, num_targets:int) -> int:
    if num_step==1:
        return num_targets
    else:
        return num_targets+recurse_uncertainty_calculation(num_step-1,num_targets=num_targets)
sum=0
for i in range(500):
    sum+=8
print(sum)