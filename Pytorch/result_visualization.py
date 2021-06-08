# We run training + validation by executing this cell

#running하면서 plot시키는 코드니까 크게 신경안써도 됨 , just train and validation 어떻게 하는지만 보기
import time
start_t = time.time()
train_losses = []
validation_losses = []
num_epochs = 720
print('STARTING TRAINING')
for i in range(num_epochs):
train_loss, train_output_seg, train_input_img = run_training(net_seg)
valid_loss, valid_output_seg, valid_input_img = run_validation(net_seg)
print('EPOCH {} of {}'.format(i, num_epochs))
print('-- train loss {} -- valid loss {} --'.format(train_loss, valid_loss))
train_losses.append(train_loss)


validation_losses.append(valid_loss)
if i%10==0:
plt.imshow(np.squeeze(valid_input_img[0, :, : , 16])) # showin
plt.show()
plt.imshow(np.squeeze(np.sum(valid_output_seg[0, :, : , :],2))) # showin
plt.show()
plt.imshow(np.squeeze(np.sum(valid_output_seg[1, :, : , :],2))) # label
plt.show()
plt.plot(range(len(train_losses)), train_losses, 'b', range(len(validation_losses)), validation_losses,'r')

red_patch = mpatches.Patch(color='red', label='Validation')
blue_patch = mpatches.Patch(color='blue', label='Training')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

torch.save(net_seg,'/content/drive/My Drive/2019_Neuro_SNloc/model/vnet_sn__multi_tmp__SN')
end_t=time.time()
print("WorkingTime: {} sec".format(end_t-start_t))
이거 나 4학년때 했던 코드인뎅
10개마다 출력되게 했었어! 돌리면서